## <font color = lightBlue> 算法模板 </font> 

### <font color = #C6E2FF> 快读 & 快写 </font> 
```cpp
int read(){
    int x = 0, f = 0; char ch = getchar();
    while(!isdigit(ch)) f |= ch=='-', ch = getchar();
    while(isdigit(ch)) x = x * 10 + (ch ^ 48), ch = getchar();
    return f ? -x : x;
}
void print(int x) {
    if(x < 0) putchar('-'), x = -x;
    if(x > 9) print(x / 10);
    putchar(x % 10 + '0');
}
```

## <font color = #C6E2FF> 基础算法 </font>

### <font color=#C6E2FF> 高精度</font> 
```cpp
高精度加法
// C = A + B, A >= 0, B >= 0
vector<int> add(vector<int> &A, vector<int> &B){
    if (A.size() < B.size()) return add(B, A);
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size(); i ++){
        t += A[i];
        if (i < B.size()) t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }
    if (t) C.push_back(t);
    return C;
}
高精度减法
// C = A - B, 满足A >= B, A >= 0, B >= 0
vector<int> sub(vector<int> &A, vector<int> &B){
    vector<int> C;
    for (int i = 0, t = 0; i < A.size(); i ++){
        t = A[i] - t;
        if (i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if (t < 0) t = 1;
        else t = 0;
    }
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
高精度乘低精度
// C = A * b, A >= 0, b >= 0
vector<int> mul(vector<int> &A, int b){
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size() || t; i ++){
        if (i < A.size()) t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
高精度除以低精度
// A / b = C ... r, A >= 0, b > 0
vector<int> div(vector<int> &A, int b, int &r){
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; i -- ){
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```

### <font color=#C6E2FF> 并查集</font> 

```cpp
int find(int x){
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}
```
### <font color = #C6E2FF> 龟速乘 </font> 
```cpp
int lpow(int x, int y){
	int res = 0;
	while(y){
		if(y & 1) res = (res + x) % mod;
		x = (x + x) % mod; y >>= 1;
	}
	return res;
}
```
### <font color = #C6E2FF> 二维前缀和 </font>  
```cpp
for(int i = 1; i <= N; i ++)
    for(int j = 1; j <= N; j ++)
        c[i][j] += c[i - 1][j] + c[i][j - 1] - c[i - 1][j - 1];
```

### <font color=#C6E2FF> st表</font> 
==O(nlog(n))==预处理，==O(1)==查询,==st(i, j)== 数组表示下标 i 开始,长度为 ==2^j^== 的区间(最大/最小值).
```cpp
int st[N][22];

void init(int n){
    for(int i = 1; i <= n; i ++) st[i][0] = a[i];
    for(int j = 1; j <= 20; j ++)
        for(int i = 1; i + (1 << j) - 1 <= n; i ++)
            st[i][j] = max(st[i][j - 1], st[i + (1 << j - 1)][j - 1]);
}
int query(int l, int r){
    if(l > r) swap(l, r);
    int k = log2(r - l + 1);
    return max(st[l][k], st[r - (1 << k) + 1][k]);
}
```

### <font color=#C6E2FF> 二分 </font> 
**返回第一个** 

```cpp
while(l < r){
    int mid = l + r >> 1;
    if(check(mid)) r = mid;
    else l = mid + 1;
}
```
**返回最后一个 ** 

```cpp
while(l < r){
    int mid = l + r + 1 >> 1;
    if(check(mid)) l = mid;
    else r = mid - 1;
}
```

**通俗的二分** 

```cpp
while(l <= r){
    int mid = l + r >> 1;
    if(a[j] / mid >= i){
        ans = mid;
        l = mid + 1;
    }
    else r = mid - 1;
}
```

**浮点二分** 

```cpp
double bsearch(double l, double r){
    double eps = 1e-7;   // eps 表示精度，取决于题目对精度的要求
    while (r - l > eps){
        double mid = (l + r) / 2;
        if (check(mid)) r = mid;
        else l = mid;
    }
    return l;
}
```
#### <font color=#C6E2FF> 求严格单调递增子序列长度 </font> 

```cpp
for(int i = 1; i <= n; i ++){
    if(a[i] > f[cnt]) f[++ cnt] = a[i];
    else{
        int l = 1, r = cnt;
        while(l < r){
            int mid = l + r >> 1;
            if(a[i] <= f[mid]) r = mid;
            else l = mid + 1;
        }
        f[l] = a[i];
    }
}
```

## <font color = #C6E2FF> DP </font> 

### <font color = #C6E2FF> 背包问题 </font> 

**背包问题的三种情况的处理方式** 

1. 体积最多是V，能获得的最大价值 ` memset(f, 0, sizeof f)` 
2. 体积恰好是V，能获得的最少价值 ` memset(f, 0x3f, sizeof f), f[0] = 0` 
3. 体积至少是V 能获得的最少价值，` memset(f, 0x3f, sizeof f), f[0] = 0, f[j] = min(f[max(0, j - v[i])], f[j])` 

### <font color = #C6E2FF> 最长公共子序列 </font> 
```cpp
for(int i = 1; i <= n; i ++)   		//第一个字符串
    for(int j = 1; j <= n; j ++){   //第二个字符串
        if(s1[i] == s2[j]) f[i][j] = f[i - 1][j - 1] + 1;
        else f[i][j] = max(f[i - 1][j], f[i][j - 1]);
    }
```
另外两个序列都是全排列的话，就可以找映射关系来求最长上升子序列。

### <font color = #C6E2FF> 树上求最长路径 </font> 
```cpp
// d1 和 d2 分别是节点向叶子节点的最长和次长距离
int dfs(int u, int fa){
    int dist = 0;
    int d1 = 0, d2 = 0;
    for(int i = head[u]; i; i = node[i].ne){
        int j = node[i].to, w = node[i].w;
        if(j == fa) continue;
        int d = dfs(j, u) + w;
        dist = max(dist, d);
        if(d >= d1) d2 = d1, d1 = d;
        else if(d >= d2) d2 = d;
    }
    ans = max(ans, d1 + d2);
    return dist;
}
```

## <font color = #C6E2FF> 数据结构 </font> 

### <font color = #C6E2FF> 并查集判二分图 </font> 

```cpp
struct DSU { //并查集模板

  vector<int> p;

  DSU(int n) : p(n + 1) { iota(p.begin(), p.end(), 0); }

  int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }

  void uni(int x, int y) { p[find(x)] = find(y); }

  bool same(int x, int y) { return find(x) == find(y); }

};

struct Edge { int u, v; } edge[M];

bool check(int n, int m) {

  DSU dsu(n * 2);

  for (int i = 1; i <= m; ++i) { //合并所有边的两个端点

    int u = edge[i].u, v = edge[i].v;

    dsu.uni(u, v + n), dsu.uni(u + n, v);

  }

  for (int i = 1; i <= n; ++i) //判断是否有i与i+n在一个集合中

    if (dsu.same(i, i + n))

      return false;

  return true;
}
```

### <font color = #C6E2FF> 单调队列 </font> 
#### <font color = #C6E2FF> 滑动窗口 </font> 
```cpp
deque<int> q;
// 求窗口内的最小值
for(int i = 1; i <= n; i ++){
    while(!q.empty() && q.front() < i - m + 1) q.pop_front();
    while(!q.empty() && a[q.back()] >= a[i]) q.pop_back();
    q.push_back(i);
    if(i >= m) cout << a[q.front()] << ' ';
}
// 求窗口内的最大值
for(int i = 1; i <= n; i ++){
    while(!q.empty() && q.front() < i - m + 1) q.pop_front();
    while(!q.empty() && a[q.back()] <= a[i]) q.pop_back();
    q.push_back(i);
    if(i >= m) cout << a[q.front()] << ' ';
}

// 常见模型：找出滑动窗口中的最大值/最小值
int q[N], hh = 0, tt = -1;
for (int i = 0; i < n; i ++ )
{
    while (hh <= tt && check_out(q[hh])) hh ++ ;  // 判断队头是否滑出窗口
    while (hh <= tt && check(q[tt], i)) tt -- ;
    q[++ tt] = i;
}
```


### <font color = #C6E2FF> KMP </font> 
```cpp
// s[]是长文本，p[]是模式串，n是s的长度，m是p的长度
// 求模式串的Next数组：
for (int i = 2, j = 0; i <= m; i ++){
    while (j && p[i] != p[j + 1]) j = ne[j];
    if (p[i] == p[j + 1]) j ++;
    ne[i] = j;
}
// 匹配
for (int i = 1, j = 0; i <= n; i ++){
    while (j && s[i] != p[j + 1]) j = ne[j];
    if (s[i] == p[j + 1]) j ++;
    if (j == m){
        j = ne[j];
    }
}
```
### <font color = #C6E2FF> add懒标记的线段树 </font> 

```cpp
struct Node{
    int l, r;
    int sum, add;
}tr[N << 2];

void pushup(int u){
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void pushdown(int u){
    Node &root = tr[u], &left = tr[u << 1], &right = tr[u << 1 | 1];
    if(root.add){
        left.add += root.add, left.sum += (left.r - left.l + 1) * root.add;
        right.add += root.add, right.sum += (right.r - right.l + 1) * root.add;
        root.add = 0;
    }
}
void build(int u, int l, int r){
    tr[u].l = l, tr[u].r = r;
    if(l == r) tr[u].sum = w[l];
    else{
        int mid = l + r >> 1;
        build(u << 1, l, mid); build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}
void modify(int u, int l, int r, int d){
    if(tr[u].l >= l && tr[u].r <= r){
        tr[u].sum += (tr[u].r - tr[u].l + 1) * d;
        tr[u].add += d;
    }
    else{
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        if(l <= mid) modify(u << 1, l, r, d);
        if(r > mid) modify(u << 1 | 1, l, r, d);
        pushup(u);
    }
}
int query(int u, int l, int r){
    if(tr[u].l >= l && tr[u].r <= r) return tr[u].sum;
    else{
        pushdown(u);
        int mid = tr[u].l + tr[u].r >> 1;
        int sum = 0;
        if(l <= mid) sum = query(u << 1, l, r);
        if(r > mid) sum += query(u << 1 | 1, l, r);
        return sum;
    }
}
```
### <font color = #C6E2FF> 树链剖分 </font> 
```
int n, idx, w[N];
int dfn[N], nw[N], top[N];
int son[N], fa[N], dep[N], sz[N];
vector<int> G[N];
struct Tree{
    int l, r;
    int add, sum;
}tr[N << 2];

void dfs1(int u, int father, int depth){
    dep[u] = depth, fa[u] = father, sz[u] = 1;
    for(auto v : G[u]){
        if(v == father) continue;
        dfs1(v, u, depth + 1);
        sz[u] += sz[v];
        if(sz[son[u]] < sz[v]) son[u] = v;
    }
}
void dfs2(int u, int t){
    dfn[u] = ++ idx, nw[idx] = w[u], top[u] = t;
    if(!son[u]) return ;

    dfs2(son[u], t);

    for(auto v : G[u]){
        if(v == fa[u] || v == son[u]) continue;

        dfs2(v, v);
    }   
}
void pushup(int u){
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}
void pushdown(int u){
    auto &root = tr[u], &left = tr[u << 1], &right = tr[u << 1 | 1];

    if(root.add){
        left.add += root.add, left.sum += (left.r - left.l + 1) * root.add;
        right.add += root.add, right.sum += (right.r - right.l + 1) * root.add;
        root.add = 0;
    }
}
void build(int u, int l, int r){
    tr[u] = {l, r, 0, nw[r]};
    if(l == r) return ;
    int mid = l + r >> 1;
    build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
    pushup(u);
}
void update(int u, int l, int r, int k){
    if(l <= tr[u].l && r >= tr[u].r){
        tr[u].add += k;
        tr[u].sum += k * (tr[u].r - tr[u].l + 1);
        return ;
    }   
    pushdown(u);
    int mid = tr[u].r + tr[u].l >> 1;
    if(l <= mid) update(u << 1, l, r, k);
    if(r > mid) update(u << 1 | 1, l, r, k);
    pushup(u);
}
int query(int u, int l, int r){
    if(l <= tr[u].l && r >= tr[u].r) return tr[u].sum;  
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    int res = 0;
    if(l <= mid) res += query(u << 1, l, r);
    if(r > mid) res += query(u << 1 | 1, l, r);
    return res;
}
void update_path(int u, int v, int k){
    while(top[u] != top[v]){
        if(dep[top[u]] < dep[top[v]]) swap(u, v);
        update(1, dfn[top[u]], dfn[u], k);
        u = fa[top[u]];
    }
    if(dep[u] < dep[v]) swap(u, v);
    update(1, dfn[v], dfn[u], k);
}
void update_tree(int u, int k){
    update(1, dfn[u], dfn[u] + sz[u] - 1, k);
}
int query_path(int u, int v){
    int res = 0;
    while(top[u] != top[v]){
        if(dep[top[u]] < dep[top[v]]) swap(u, v);
        res += query(1, dfn[top[u]], dfn[u]);
        u = fa[top[u]];
    }
    if(dep[u] < dep[v]) swap(u, v);
    res += query(1, dfn[v], dfn[u]);
    return res;
}
int query_tree(int u){
    return query(1, dfn[u], dfn[u] + sz[u] - 1);
}
```


### <font color = #C6E2FF> 树状数组  </font> 

```cpp
void update(int i, int x){
	while(i <= n){
		c[i] += x;
		i += lowbit(i);
	}
}
int sum(int i){
	int res = 0;
	while(i >= 1){
		res += c[i];
		i -= lowbit(i);
	}
	return res;
}
```
### <font color = #C6E2FF> 字符串哈希 </font> 
核心思想：将字符串看成P进制数，P的进值是131或13331，取这两个值的冲突概率低(0.01%)
小技巧：取模的数用 ==2^64^==，这样直接用==unsigned long long==存储，溢出的结果就是取模的结果​ 

```cpp
typedef unsigned long long ull;
ull h[N], p[N], P = 131; // h[k]存储字符串前k个字母的哈希值, p[k]存储 P^k mod 2^64

// 初始化
p[0] = 1;
for (int i = 1; i <= n; i ++){
    h[i] = h[i - 1] * P + str[i];
    p[i] = p[i - 1] * P;
}

// 计算子串 str[l ~ r] 的哈希值
ull get(int l, int r){
    return h[r] - h[l - 1] * p[r - l + 1];
}
```
## <font color = #C6E2FF> 图论 </font> 

### <font color = #C6E2FF> LCA 求最近公共祖先 </font> 

<img src="C:\Users\86152\AppData\Roaming\Typora\typora-user-images\image-20220712165553072.png" alt="image-20220712165553072" style="zoom:67%;" />

```cpp
int depth[N], fa[N][21];

void bfs(int root){
    queue<int> q;
    q.push(root);
    memset(depth, 0x3f, sizeof depth);
    depth[0] = 0, depth[root] = 1;
    while(!q.empty()){
        int t = q.front(); q.pop();
        for(int i = h[t]; i; i = node[i].ne){
            int j = node[i].to;
            if(depth[j] > depth[t]){
                depth[j] = depth[t] + 1;
                q.push(j);
                fa[j][0] = t;
                for (int k = 1; k <= 19; k ++)
                    fa[j][k] = fa[fa[j][k - 1]][k - 1];
            }
        }
    }
}
int lca(int a, int b){
    if(depth[a] < depth[b]) swap(a, b);
    for(int k = 19; k >= 0; k --)
        if(depth[fa[a][k]] >= depth[b])
            a = fa[a][k];
    if(a == b) return a;
    for(int k = 19; k >= 0; k --)
        if(fa[a][k] != fa[b][k])
            a = fa[a][k], b = fa[b][k];
    return fa[a][0];
}
```
### <font color = #C6E2FF> dijkstra </font> 

```cpp
int dis[N], h[N], idx;
bool vis[N];
struct Edge{
	int to, w, ne;
}edge[M];
struct Node{
	int u, dis;
    friend bool operator < (Node x, Node y){
        return x.dis > y.dis;
    }
};
priority_queue<Node> q;

void add(int u, int v, int w){
    Edge &t = edge[++ idx];
	t.ne = h[u], t.to = v, t.w = w, h[u] = idx;
}
void dij(int s){
    memset(dis, 0x3f, sizeof dis);
    dis[s] = 0;
	q.push((Node){s, 0});
	while(!q.empty()){
		Node cur = q.top(); q.pop();
        
		if(vis[cur.u]) continue;
		vis[cur.u] = true;
        
		for(int i = h[cur.u]; i; i = edge[i].ne){
			int v = edge[i].to, w = edge[i].w;
            
			if(!vis[v] && dis[cur.u] + w < dis[v]){
				dis[v] = dis[cur.u] + w;
				q.push((Node){v, dis[v]});
			}
		}
	}
}
```
### <font color = #C6E2FF> spfa </font> 
```cpp
int dis[N], st[N];

void spfa(int S){
	memset(dis, 0x3f, sizeof dis);
	dis[S] = 0;
	queue<int> q;
	q.push(S); st[S] = true;
	while(!q.empty()){
		int t = q.front(); q.pop();
		st[t] = false;
		for(int i = h[t]; i; i = node[i].ne){
			int j = node[i].to, w = node[i].w;
			if(dis[j] > dis[t] + w){
				dis[j] = dis[t] + w;
				if(!st[j]){
                    q.push(j);
					st[j] = true;
				}
			}
		}
	}
}
```

## <font color = #C6E2FF> 数学 </font> 

### <font color = #C6E2FF> 组合数 </font> 

**预处理逆元求组合数** 

```cpp
int fact[N], infact[N];

int qmi(int a, int b, int p){
    int res = 1 % p; a %= p;
    while (b > 0){
        if(b & 1) res = res * a % p;
        a = a * a % p; b >>= 1;
    }
    return res;
}

void init(int n){
    fact[0] = infact[0] = 1;
    for (int i = 1; i <= n; i ++){
        fact[i] = fact[i - 1] * i % mod;
        infact[i] = qmi(fact[i], mod - 2, mod);
    }
}
int C(int a, int b){
    //[ use init() ] and [open long long]
    if (a < b) return 0;
    return fact[a] * infact[b] % mod * infact[a - b] % mod;
}
int P(int a, int b){
    if(a < b) return 0;
    return fact[a] * infact[a - b] % mod;
}
```

**递推求组合数** 

```cpp
int c[N][N];
for (int i = 0; i < N; i ++)
    for (int j = 0; j <= i; j ++){
        if (!j) c[i][j] = 1;
        else c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % mod;
    }
```
一个底和高分别为a，b的一个直角三角形，他的斜边经过的整数点的个数是 **__gcd(a, b) + 1** 

### <font color = #C6E2FF> 矩阵快速幂 </font> 

```cpp
struct mat{
    long long A[N][N] = {0};
    void build(){  //建立单位矩阵
        rep(i, 1, N) A[i][i] = 1;
    }
    mat const operator * (mat B) const{
        mat C;
        rep(i, 1, N) rep(j, 1, N) rep(k, 1, N)
            C.A[i][j] = mod(C.A[i][j] + mod(A[i][k] * B.A[k][j]));
        return C;
    }
};
```


### <font color = #C6E2FF> 卡特兰数 </font> 
```cpp
给定n个0和n个1，它们按照某种顺序排成长度为2n的序列，满足任意前缀中0的个数都不少于1的个数的序列的数量为： Cat(n) = C(2n, n) / (n + 1)
```

### <font color = #C6E2FF> 整除分块 </font> 
```cpp
long long H(int n) {
    long long res = 0;  // 储存结果
    int l = 1, r;       // 块左端点与右端点
    while (l <= n) {
        r = n / (n / l);  // 计算当前块的右端点
        res += (r - l + 1) * 1ll * (n / l);  // 累加这一块的贡献到结果中。乘上 1LL 防止溢出
        l = r + 1;  // 左端点移到下一块
    }
    return res;
}
```

### <font color = #C6E2FF> 判凸包 </font> 

```cpp
int n, stk[N], top;
bool st[N];
struct Point {
    double x, y;
    bool operator < (const Point &t) const {
        if(x != t.x) return x < t.x;
        return y < t.y;
    }
    Point operator - (const Point &t) const {
        return (Point){x - t.x, y - t.y};
    }
};
Point q[N];

double get_dist(Point a, Point b)
{
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

double cross(Point a, Point b)
{
    return a.x * b.y - a.y * b.x;
}

double area(Point a, Point b, Point c)
{
    return cross(b - a, c - a);
}

bool andrew()
{
    sort(q, q + n);
    top = 0;
    for(int i = 0 ; i < n; i ++)
    {
        while(top >= 2 && area(q[stk[top - 1]], q[stk[top]], q[i]) <= 0)
        {
            if(area(q[stk[top - 1]], q[stk[top]], q[i]) < 0)
                st[stk[top -- ]] = false;
            else top --;
        }
        stk[++ top] = i;
        st[i] = true;
    }

    st[0] = false;
    for(int i = n - 1; i >= 0; i --)
    {
        if(st[i]) continue;
        while(top >= 2 && area(q[stk[top - 1]], q[stk[top]], q[i]) <= 0)
            top --;
        stk[++ top] = i;
    }

	return top == 5;
}
```

## <font color = #C6E2FF> 小知识 </font> 

### <font color = #C6E2FF> cf的保护分 </font>
```cpp
前六场初始分：Promotions of the displayed rating will be equal to 500,350,250,150,100,50 (in total exactly 1400).
```

### <font color = #C6E2FF> 数据范围所能使用的时间复杂度 </font> 

  ![](https://pic1.imgdb.cn/item/634775fe16f2c2beb1ec7038.png) 

### <font color=#C6E2FF> cf 防止哈希被卡</font> 
```cpp
struct custom_hash {
	static uint64_t splitmix64(uint64_t x) {
		x += 0x9e3779b97f4a7c15;
		x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
		x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
		return x ^ (x >> 31);
	}
	size_t operator()(uint64_t x) const {
		static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
		return splitmix64(x + FIXED_RANDOM);
	}
};
unordered_map<int, int, custom_hash> safe_map;
//参考链接https://codeforces.com/blog/entry/62393
```

### <font color=#C6E2FF> 能用PII作哈希的键值</font> 
```cpp
struct hashfunc{
    template<typename T, typename U>
    size_t operator() (const pair<T, U> &i) const{
        return hash<T>()(i.first) ^ hash<U>()(i.second);
    }
};
```
### <font color=#C6E2FF> Debug</font> 
``` cpp
string to_string(string s) { return '"' + s + '"'; }
string to_string(char s) { return string(1, s); }
string to_string(const char* s) { return to_string((string)s); }
string to_string(bool b) { return (b ? "true" : "false"); }

template <typename A>
string to_string(A);

template <typename A, typename B>
string to_string(pair<A, B> p) { return "(" + to_string(p.first) + ", " + to_string(p.second) + ")"; }

template <typename A>
string to_string(A v) {
    bool f = 1;
    string r = "{";
    for (const auto& x : v) {
        if (!f) r += ", ";
        f = 0;
        r += to_string(x);
    }
    return r + "}";
}

void debug_out() { cout << endl; }
template <typename Head, typename... Tail>
void debug_out(Head H, Tail... T) {
    cout << " " << to_string(H);
    debug_out(T...);
}
#define pr(...) cout << "[" << #__VA_ARGS__ << "] :", debug_out(__VA_ARGS__)

#define dearr(arr, a, b)                                \
    cout << #arr << " : ";                              \
    for (int i = a; i <= b; i++) cout << arr[i] << " "; \
    cout << endl;

#define demat(mat, row, col)                                     \
    cout << #mat << " :\n";                                      \
    for (int i = 1; i <= row; i++) {                             \
        for (int j = 1; j <= col; j++) cout << mat[i][j] << " "; \
        cout << endl;                                            \
    }

void debit(int x){
    int t = log2(x);
    vector<int> v;
    for(int i = t; i >= 0; i --){
        if(x >> i & 1) v.push_back(1);
        else v.push_back(0);
    }
    for(auto bit : v) cout << bit; cout << endl;
}
```
### <font color=#C6E2FF> 取模int </font> 

```cpp
// assume -P <= x < 2P
int norm(int x) {
    if (x < 0) {
        x += P;
    }
    if (x >= P) {
        x -= P;
    }
    return x;
}
template<class T>
T power(T a, int b) {
    T res = 1;
    for (; b; b /= 2, a *= a) {
        if (b % 2) {
            res *= a;
        }
    }
    return res;
}
struct Z {
    int x;
    Z(int x = 0) : x(norm(x)) {}
    int val() const {
        return x;
    }
    Z operator-() const {
        return Z(norm(P - x));
    }
    Z inv() const {
        assert(x != 0);
        return power(*this, P - 2);
    }
    Z &operator*=(const Z &rhs) {
        x = ll(x) * rhs.x % P;
        return *this;
    }
    Z &operator+=(const Z &rhs) {
        x = norm(x + rhs.x);
        return *this;
    }
    Z &operator-=(const Z &rhs) {
        x = norm(x - rhs.x);
        return *this;
    }
    Z &operator/=(const Z &rhs) {
        return *this *= rhs.inv();
    }
    friend Z operator*(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res *= rhs;
        return res;
    }
    friend Z operator+(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res += rhs;
        return res;
    }
    friend Z operator-(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res -= rhs;
        return res;
    }
    friend Z operator/(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res /= rhs;
        return res;
    }
};
```

### <font color = #C6E2FF>  火车头 </font>
```cpp
#pragma GCC optimize(2)
#pragma GCC optimize(3)
#pragma GCC optimize("Ofast")
#pragma GCC optimize("inline")
#pragma GCC optimize("-fgcse")
#pragma GCC optimize("-fgcse-lm")
#pragma GCC optimize("-fipa-sra")
#pragma GCC optimize("-ftree-pre")
#pragma GCC optimize("-ftree-vrp")
#pragma GCC optimize("-fpeephole2")
#pragma GCC optimize("-ffast-math")
#pragma GCC optimize("-fsched-spec")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("-falign-jumps")
#pragma GCC optimize("-falign-loops")
#pragma GCC optimize("-falign-labels")
#pragma GCC optimize("-fdevirtualize")
#pragma GCC optimize("-fcaller-saves")
#pragma GCC optimize("-fcrossjumping")
#pragma GCC optimize("-fthread-jumps")
#pragma GCC optimize("-funroll-loops")
#pragma GCC optimize("-fwhole-program")
#pragma GCC optimize("-freorder-blocks")
#pragma GCC optimize("-fschedule-insns")
#pragma GCC optimize("inline-functions")
#pragma GCC optimize("-ftree-tail-merge")
#pragma GCC optimize("-fschedule-insns2")
#pragma GCC optimize("-fstrict-aliasing")
#pragma GCC optimize("-fstrict-overflow")
#pragma GCC optimize("-falign-functions")
#pragma GCC optimize("-fcse-skip-blocks")
#pragma GCC optimize("-fcse-follow-jumps")
#pragma GCC optimize("-fsched-interblock")
#pragma GCC optimize("-fpartial-inlining")
#pragma GCC optimize("no-stack-protector")
#pragma GCC optimize("-freorder-functions")
#pragma GCC optimize("-findirect-inlining")
#pragma GCC optimize("-fhoist-adjacent-loads")
#pragma GCC optimize("-frerun-cse-after-loop")
#pragma GCC optimize("inline-small-functions")
#pragma GCC optimize("-finline-small-functions")
#pragma GCC optimize("-ftree-switch-conversion")
#pragma GCC optimize("-foptimize-sibling-calls")
#pragma GCC optimize("-fexpensive-optimizations")
#pragma GCC optimize("-funsafe-loop-optimizations")
#pragma GCC optimize("inline-functions-called-once")
#pragma GCC optimize("-fdelete-null-pointer-checks")
```

### <font color=#C6E2FF> C++ STL简介</font> 
```cpp
priority_queue, 优先队列，默认是大顶堆
    size()
    empty()
    push()  插入一个元素
    top()  返回堆顶元素
    pop()  弹出堆顶元素
    定义成小顶堆的方式：priority_queue<int, vector<int>, greater<int>> q;

stack, 栈
    size()
    empty()
    push()  向栈顶插入一个元素
    top()  返回栈顶元素
    pop()  弹出栈顶元素

deque, 双端队列
    size()
    empty()
    clear()
    front()/back()
    push_back()/pop_back()
    push_front()/pop_front()
    begin()/end()
    []

bitset, 
    bitset<10000> s;
    ~, &, |, ^
    >>, <<
    ==, !=
    []

    count()  返回有多少个1

    any()  判断是否至少有一个1
    none()  判断是否全为0

    set()  把所有位置成1
    set(k, v)  将第k位变成v
    reset()  把所有位变成0
    flip()  等价于~
    flip(k) 把第k位取反
```
