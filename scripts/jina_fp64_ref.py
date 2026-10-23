import sys, json, mmap, glob, numpy as np; sys.path.insert(0,"scripts"); sys.path.insert(0,".")
import jina_q80_eval as J, urllib.request
p=glob.glob("/home/lovedheart/.cache/huggingface/hub/models--jinaai--jina-embeddings-v5-text-small-retrieval/snapshots/*/model.safetensors")[0]
_fh=open(p,"rb"); _mm=mmap.mmap(_fh.fileno(),0,access=mmap.ACCESS_READ)
_hl=int.from_bytes(_mm[:8],"little"); _hdr=json.loads(_mm[8:8+_hl])
def get(k):
    m=_hdr[k]; s_,e_=m["data_offsets"]
    u=np.frombuffer(_mm,np.uint16,(e_-s_)//2,8+_hl+s_)
    return (u.astype(np.uint32)<<16).view(np.float32).astype(np.float64).reshape(m["shape"])
theta=3.5e6; inv=1.0/theta**(np.arange(0,128,2)/128.0)
def rope(x,pos):  # rotate_half / llama-style
    ang=pos[:,None,None].astype(np.float64)*inv[None,None,:]
    cos=np.concatenate([np.cos(ang)]*2,-1); sin=np.concatenate([np.sin(ang)]*2,-1)
    x1,x2=x[...,:64],x[...,64:]
    rot=np.concatenate([-x2,x1],-1)
    return x*cos+rot*sin
emb=get("embed_tokens.weight"); on=get("norm.weight")
def fwd(ids):
    x=emb[np.asarray(ids)]; T=len(ids); pos=np.arange(T); eps=1e-6
    causal=np.triu(np.full((1,T,T),-np.inf),1)
    for l in range(28):
        pre=f"layers.{l}."; h=x
        xr=x/np.sqrt((x*x).mean(-1,keepdims=True)+eps)*get(pre+"input_layernorm.weight")
        q=(xr@get(pre+"self_attn.q_proj.weight").T).reshape(T,16,128)
        k=(xr@get(pre+"self_attn.k_proj.weight").T).reshape(T,8,128)
        v=(xr@get(pre+"self_attn.v_proj.weight").T).reshape(T,8,128)
        q=get(pre+"self_attn.q_norm.weight")*(q/np.sqrt((q*q).mean(-1,keepdims=True)+eps))
        k=get(pre+"self_attn.k_norm.weight")*(k/np.sqrt((k*k).mean(-1,keepdims=True)+eps))
        q,k=rope(q,pos),rope(k,pos); k=np.repeat(k,2,1); v=np.repeat(v,2,1)
        s=q.transpose(1,0,2)@k.transpose(1,2,0)/np.sqrt(128)+causal
        s-=s.max(-1,keepdims=True); pr=np.exp(s); pr/=pr.sum(-1,keepdims=True)
        o=(pr@v.transpose(1,0,2)).transpose(1,0,2).reshape(T,2048)
        x=h+o@get(pre+"self_attn.o_proj.weight").T
        xr=x/np.sqrt((x*x).mean(-1,keepdims=True)+eps)*get(pre+"post_attention_layernorm.weight")
        g=xr@get(pre+"mlp.gate_proj.weight").T; u=xr@get(pre+"mlp.up_proj.weight").T
        x=x+(g/(1+np.exp(-g))*u)@get(pre+"mlp.down_proj.weight").T
    x=x/np.sqrt((x*x).mean(-1,keepdims=True)+eps)*on
    e=x[-1]; return e/np.linalg.norm(e)
if __name__=="__main__":
    texts = ["A soccer game with multiple males playing.","Some men are playing a sport.",
     "A man playing the guitar.","The guitar is being played by a woman.",
     "A woman slicing bread with a knife.","The bread was cut with a sharp knife.",
     "The stock market rallied on strong earnings.","Investors celebrated as equities climbed sharply.",
     "The cat sat on the mat and purred.","A feline rested on a rug and made a rumbling sound.",
     "Comets are icy bodies from the outer solar system.","Garbage trucks collect household waste every week.",
     "She typed the report before the meeting began.","The deadline for the report was moved to Friday.",
     "Ocean temperatures are rising due to greenhouse gases.","Global warming is warming the seas."]
    R=J.remote(8999,texts); R/=np.linalg.norm(R,axis=1,keepdims=True)
    j=J.Jina(); ids=[j.tok.encode(t).ids for t in texts]
    F=np.stack([fwd(i) for i in ids]); F/=np.linalg.norm(F,axis=1,keepdims=True)
    d=(F*R).sum(1); print("rotate_half fp64 vs vLLM:",np.round(d,5),"min",d.min())
    Q=np.stack([j.forward(i) for i in ids]); Q/=np.linalg.norm(Q,axis=1,keepdims=True)
    print("q8 vs fp64(rh):",np.round((Q*F).sum(1),5))
    sf,sr=F@F.T,R@R.T
    print("max|sim diff|",np.abs(sf-sr).max())
    def nn(S):
        out=[]
        for i in range(S.shape[0]):
            r=S[i].copy(); r[i]=-9; out.append(int(np.argmax(r)))
        return out
    print("NN fp64 ",nn(sf)); print("NN vllm ",nn(sr))
