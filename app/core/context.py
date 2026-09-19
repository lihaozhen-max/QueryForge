import asyncio
from _contextvars import ContextVar,Token
from dataclasses import dataclass


request_context_var = ContextVar('req_id', default = '')

def set_request_id(req_id:str)->Token:
    return request_context_var.set(req_id)

def get_request_id():
    return  request_context_var.get()

def reset_request_id(token:Token):
    request_context_var.reset(token)

if __name__ == '__main__':
    async def req1():
        req_id = get_request_id()
        print(f"-1-req1 req_id={req_id}")  # ""
        token = set_request_id("1111")
        await asyncio.sleep(1)
        print(f"-2-req1 req_id={get_request_id()}")  # "1111"
        await asyncio.sleep(1)
        reset_request_id(token)
        print(f"-3-req1 req_id={get_request_id()}")  # ""


    async def req2():
        req_id = get_request_id()
        print(f"-1-req2 req_id={req_id}")  # ""
        token = set_request_id("2222")
        await asyncio.sleep(1)
        print(f"-2-req2 req_id={get_request_id()}")  # "222"
        await asyncio.sleep(1)
        # reset_request_id(token)
        print(f"-3-req2 req_id={get_request_id()}")  # ""

    async def test():
        await asyncio.gather(req1(), req2())

    asyncio.run(test())