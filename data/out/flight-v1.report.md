# flight-v1 extraction report

Units: 1072  Edges: 26544 (w >= 5)  Types: 290

## Units by role

role
input     359
vnc       298
brain     291
output     83
dn         41

## Transmitter sign by role

nt      acetylcholine  gaba  glutamate  histamine  unclear
role                                                      
brain             213    62         16          0        0
dn                 41     0          0          0        0
input             355     4          0          0        0
output              0     0         50          0       33
vnc               161   108         25          2        2

## Static check 1: what drives DNg02 (top pre-types, summed synapses)

pre                   post   
PS080                 DNg02_c    221
IN06B013              DNg02_f    159
PS126                 DNg02_a    113
IN03B043              DNg02_a    109
IN06B013              DNg02_e     95
                      DNg02_d     56
PS080                 DNg02_b     49
IN03B043              DNg02_c     45
vCal3                 DNg02_a     44
AN19B017              DNg02_a     42
PS311                 DNg02_d     40
PS080                 DNg02_e     36
CL053                 DNg02_g     30
DNbe001               DNg02_a     29
CB0285                DNg02_b     22
CL053                 DNg02_b     20
AN23B001              DNg02_c     18
WED071                DNg02_d     15
IN06B013              DNg02_c     15
IN06B055              DNg02_d     14
LAL084                DNg02_b     14
CL053                 DNg02_a     12
untyped_cb_intrinsic  DNg02_f     10
AN19B059              DNg02_d     10
IN00A043              DNg02_g      9

## Static check 2: what DNg02 drives (top post-types)

pre      post     
DNg02_a  MNwm36       577
         IN19B043     522
DNg02_g  MNwm36       477
DNg02_b  tp2 MN       473
DNg02_a  tp2 MN       453
DNg02_c  MNwm36       358
DNg02_a  IN27X014     354
         dMS10        325
DNg02_b  AN27X008     304
         MNwm36       281
DNg02_c  ps1 MN       267
DNg02_g  tp2 MN       266
DNg02_e  MNwm36       263
DNg02_g  AN27X008     236
DNg02_f  DLMn c-f     218
DNg02_d  MNwm36       204
DNg02_f  MNwm36       189
DNg02_e  ps1 MN       166
DNg02_g  INXXX146     158
         IN19B043     158
DNg02_a  IN06A039     146
DNg02_f  IN11B014     139
         ps1 MN       139
DNg02_g  IN07B030     129
DNg02_e  DVMn 1a-c    129

## Static check 3: DNg02 laterality onto wing/haltere MNs

(DN soma side, MN soma side, path) -> synapse weight (via-vnc is w1*w2/100)

- ('L', 'L', 'direct'): 1773
- ('L', 'L', 'via-vnc'): 9447
- ('L', 'R', 'direct'): 1476
- ('L', 'R', 'via-vnc'): 9464
- ('R', 'L', 'direct'): 1424
- ('R', 'L', 'via-vnc'): 9002
- ('R', 'R', 'direct'): 1426
- ('R', 'R', 'via-vnc'): 8856

## Static check 4: LPTC -> DN direct edges

pre    post 
LC4    DNp04    11597
       DNp01     6362
LPLC2  DNp01     4836
LC4    DNp02     4205
LPLC2  DNp04     3381
LC4    DNp03     2492
LPLC2  DNp06     1581
LC4    DNp05     1255
       DNp06     1098

## Wing / haltere MNs present

somaSide    L  R
type            
            0  1
DLMn a, b   1  1
DLMn c-f    4  4
DVMn 1a-c   3  3
DVMn 2a, b  2  2
DVMn 3a, b  2  2
MNhm03      1  1
MNhm42      1  1
MNhm43      1  1
MNwm35      1  1
MNwm36      1  1
STTMm       2  2
TTMn        1  1
b1 MN       1  1
b2 MN       1  1
b3 MN       1  1
hDVM MN     1  1
hg1 MN      1  1
hg2 MN      1  1
hg3 MN      1  1
hg4 MN      1  1
hi1 MN      1  1
hi2 MN      2  2
hiii2 MN    1  1
i1 MN       1  1
i2 MN       1  1
iii1 MN     1  1
iii3 MN     1  1
ps1 MN      1  1
ps2 MN      1  1
tp1 MN      1  1
tp2 MN      1  1
tpn MN      1  1
