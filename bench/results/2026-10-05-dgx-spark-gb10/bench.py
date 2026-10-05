import json, urllib.request, random, time, sys
URL="http://127.0.0.1:8081/v1/chat/completions"
random.seed(1)
words="the of and system memory expert cache token layer model graph kernel stream value queue buffer window router weight scale block chunk".split()
def post(body):
    r=urllib.request.Request(URL,json.dumps(body).encode(),{"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=3600))
def run(target, gen=128):
    n=int(target*0.75)   # ~1.33 tokens per word of this vocabulary mix
    text=f"[run {random.random()}] "+" ".join(random.choice(words) for _ in range(n))
    body={"model":"strata","max_tokens":gen,"temperature":0,"reasoning_effort":"none",
          "messages":[{"role":"user","content":text+"\n\nWrite a short story about a robot learning to paint."}]}
    t=time.time(); d=post(body); wall=time.time()-t
    tm=d["timings"]; u=d["usage"]
    return (u["prompt_tokens"], tm["prompt_per_second"], tm["predicted_n"], tm["predicted_per_second"], wall)
print(f"{'prompt_tok':>10} {'prefill tok/s':>14} {'gen_n':>6} {'decode tok/s':>13} {'wall s':>8}", flush=True)
run(200,16)  # warm-up
for c in [512,2048,8192,16384,32768,65536,100000]:
    try:
        p,pp,gn,gp,w=run(c)
        print(f"{p:>10} {pp:>14.1f} {gn:>6} {gp:>13.1f} {w:>8.1f}", flush=True)
    except Exception as e:
        print(c,"failed:",e, flush=True); break
