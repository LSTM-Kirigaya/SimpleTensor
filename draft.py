

def dump_ans(array, length):
    if len(array) < length:
        return 0
    ans = 0
    depth = -1 * min(array)
    for layer in range(depth):
        cur_length = 0
        for i in range(len(array)):
            if array[i] >= (- layer):
                ans += cur_length // length
                cur_length = 0
            else:
                cur_length += 1

        if cur_length > 0:
            ans += cur_length // length

    return ans

def func():
    N = int(input())
    _ = int(input())
    arrays = input().split(',')
    arrays = list(map(int, arrays))
    if N > len(arrays):
        print(0)
        return
    for i in range(len(arrays)):
        arrays[i] = min(0, arrays[i])
    ans = 0
    buffer = []
    for a in arrays:
        if a == 0:
            if len(buffer) > 0:
                ans += dump_ans(buffer, N)
                buffer = []
        else:
            buffer.append(a)
    if len(buffer) > 0:
        ans += dump_ans(buffer, N)
    print(ans)


if __name__ == "__main__":
    func()
