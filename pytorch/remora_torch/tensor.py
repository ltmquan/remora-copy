"""Name-independent optimizations over typed, specialized Remora expressions.

Match operations and parameter identities, never source-library names/hashes.
Floating reductions may reassociate; explicit static/general modes stay ordered.
"""
from pathlib import Path
from .compiler import Generator, CompileError, key, typ, shape, cell_shapes, spec


def application(expr):
    args=[]
    while expr['tag']=='App':args.append(expr['arg']);expr=expr['fun']
    return expr,list(reversed(args))


class TensorGenerator(Generator):
    def __init__(self,ast):
        super().__init__(ast)
        self.optimizations=[]
        self.bound_ids=set()
        def collect(node):
            if isinstance(node,dict):
                if isinstance(node.get('var'),dict) and 'varName' in node['var']:
                    self.bound_ids.add(key(node['var']))
                for value in node.values():collect(value)
            elif isinstance(node,list):
                for value in node:collect(value)
        collect(ast)

    def builtin(self,expr):
        if expr['tag']=='Var' and key(expr['name']) not in self.bound_ids:
            return expr['name']['varName']
        return None

    def combiner(self,expr):
        direct=self.builtin(expr)
        if direct in ('+','f.+'):return 'sum'
        # Recognize an eta-expanded add regardless of the function's name.
        if expr['tag']=='Var':
            b=self.defs.get(key(expr['name']))
            if b and b['tag']=='BindFun' and len(b['params'])==2:
                head,args=application(b['expr'])
                if self.builtin(head) in ('+','f.+') and len(args)==2:
                    if all(x['tag']=='Var' for x in args) and {key(x['name']) for x in args}=={key(p['var']) for p in b['params']}:
                        return 'sum'
        return None

    def reduction(self,expr):
        head,args=application(expr);name=self.builtin(head)
        if name=='reduce' and len(args)==2 and self.combiner(args[0]):return args[1],None
        if name=='reduce/zero' and len(args)==3 and self.combiner(args[0]):return args[2],args[1]
        return None

    def is_zero(self,x):
        return x is None or (x['tag']=='Array' and not x['dims'] and len(x['atoms'])==1 and x['atoms'][0]['tag']=='Base' and x['atoms'][0]['lit']['lit']==0)

    def dot_parameters(self,fn):
        if fn['tag']!='Var':return None
        b=self.defs.get(key(fn['name']))
        if not b or b['tag']!='BindFun' or len(b['params'])!=2:return None
        if any(typ(p)['arrayTypeAtom']['tag']!='Float' or len(cell_shapes(typ(p)))!=1 for p in b['params']):return None
        reduction=self.reduction(b['expr'])
        if not reduction or not self.is_zero(reduction[1]):return None
        head,args=application(reduction[0])
        if self.builtin(head)!='f.*' or len(args)!=2 or any(x['tag']!='Var' for x in args):return None
        ids=[key(p['var']) for p in b['params']]
        if sorted(key(x['name']) for x in args)!=sorted(ids):return None
        return b

    def matrix_pattern(self,head,args):
        if head['tag']!='Var':return None
        b=self.defs.get(key(head['name']))
        if not b or b['tag']!='BindFun' or len(b['params'])!=2 or len(args)!=2:return None
        inner,inner_args=application(b['expr'])
        if not self.dot_parameters(inner) or len(inner_args)!=2 or any(v['tag']!='Var' for v in inner_args):return None
        ids={key(p['var']):i for i,p in enumerate(b['params'])}
        if set(ids)!=set(key(v['name']) for v in inner_args):return None
        cells=[cell_shapes(typ(p)) for p in b['params']]
        if sorted(map(len,cells))!=[1,2]:return None
        vector=next(i for i,c in enumerate(cells) if len(c)==1);matrix=1-vector
        actual=[cell_shapes(typ(x)) for x in args]
        if len(actual[matrix])!=2 or not actual[vector] or actual[vector][-1]!=actual[matrix][-1]:return None
        return vector,matrix

    def expr(self,e,lines,env):
        if e['tag']=='App':
            head,args=application(e)
            pattern=self.matrix_pattern(head,args)
            if pattern:
                # Evaluate actuals once, in source order, then a whole-array GEMM.
                values=[self.expr(x,lines,env) for x in args];vector,matrix=pattern
                self.optimizations.append('matmul')
                return self.emit(lines,f'torch.matmul({values[vector]}, {values[matrix]}.transpose(-1, -2))')
            if self.dot_parameters(head) and len(args)==2:
                shapes=[cell_shapes(typ(x)) for x in args]
                values=[self.expr(x,lines,env) for x in args]
                self.optimizations.append('dot')
                return self.emit(lines,f'_tensor_dot({values[0]}, {values[1]}, {shapes!r})')
            reduction=self.reduction(e)
            if reduction:
                data,zero=reduction
                # Restrict to fully applied, scalar function reductions with no lifted combiner.
                if not shape(typ(args[0])['arrayTypeShape']) and typ(data)['arrayTypeAtom']['tag'] in ('Float','Int'):
                    fn_type=typ(head)['arrayTypeAtom']
                    cells=[]
                    for _ in args:
                        cells.append(cell_shapes(fn_type['in']));fn_type=fn_type['out']['arrayTypeAtom']
                    n=len(cells[-1]);axis=len(cell_shapes(typ(data)))-n
                    if axis<0 or n<1:return super().expr(e,lines,env)
                    # Initializer is evaluated before the data, as in the original call.
                    z=self.expr(zero,lines,env) if zero is not None else None
                    x=self.expr(data,lines,env)
                    self.optimizations.append('sum')
                    reduced=self.emit(lines,f'torch.sum({x}, dim={axis}, dtype={x}.dtype)')
                    if z is None:return reduced
                    # Prefix agreement applies to zero's frame as well as the array frame.
                    return self.emit(lines,f'_tensor_add_initial({reduced}, {z}, {cell_shapes(typ(e))!r}, {cells[1]!r})')
        return super().expr(e,lines,env)

    def generate(self):
        code=super().generate()
        runtime=Path(__file__).with_name('tensor_runtime.py').read_text()
        code=code.replace("LOWERING = 'static'", "LOWERING = 'tensor'\nTENSOR_OPS = "+repr(self.optimizations)+"\nNUMERICS = 'native reductions; reassociation permitted'")
        return runtime+'\n'+code


def compile_tensor_ast(ast,require_optimization=False):
    if ast.get('tag')!='Prog':
        ast={'tag':'Prog','progDecs':[{'tag':'Entry','var':{'varName':'main','varTag':{'getTag':-1}},'params':[],'expr':ast}]}
    generator=TensorGenerator(ast);code=generator.generate()
    if require_optimization and not generator.optimizations:raise CompileError('no structural tensor optimization')
    return code
