
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(ms3ggt_0,156,482). coords(ms3ggt_0,78,179,482).  visible(ms3ggt_0,482). happensAt(walking(ms3ggt_0),482). orientation(ms3ggt_1,14,482). coords(ms3ggt_1,249,228,482).  visible(ms3ggt_1,482). happensAt(walking(ms3ggt_1),482).
orientation(ms3ggt_0,156,483). coords(ms3ggt_0,78,181,483).  visible(ms3ggt_0,483). happensAt(walking(ms3ggt_0),483). orientation(ms3ggt_1,14,483). coords(ms3ggt_1,250,231,483).  visible(ms3ggt_1,483). happensAt(walking(ms3ggt_1),483).
orientation(ms3ggt_0,156,484). coords(ms3ggt_0,78,181,484).  visible(ms3ggt_0,484). happensAt(walking(ms3ggt_0),484). orientation(ms3ggt_1,14,484). coords(ms3ggt_1,250,235,484).  visible(ms3ggt_1,484). happensAt(walking(ms3ggt_1),484).
orientation(ms3ggt_0,156,485). coords(ms3ggt_0,78,182,485).  visible(ms3ggt_0,485). happensAt(walking(ms3ggt_0),485). orientation(ms3ggt_1,18,485). coords(ms3ggt_1,251,238,485).  visible(ms3ggt_1,485). happensAt(walking(ms3ggt_1),485).
orientation(ms3ggt_0,156,486). coords(ms3ggt_0,77,182,486).  visible(ms3ggt_0,486). happensAt(walking(ms3ggt_0),486). orientation(ms3ggt_1,18,486). coords(ms3ggt_1,252,240,486).  visible(ms3ggt_1,486). happensAt(walking(ms3ggt_1),486).
orientation(ms3ggt_0,156,487). coords(ms3ggt_0,77,182,487).  visible(ms3ggt_0,487). happensAt(walking(ms3ggt_0),487). orientation(ms3ggt_1,22,487). coords(ms3ggt_1,253,242,487).  visible(ms3ggt_1,487). happensAt(walking(ms3ggt_1),487).
orientation(ms3ggt_0,160,488). coords(ms3ggt_0,76,183,488).  visible(ms3ggt_0,488). happensAt(walking(ms3ggt_0),488). orientation(ms3ggt_1,22,488). coords(ms3ggt_1,255,244,488).  visible(ms3ggt_1,488). happensAt(walking(ms3ggt_1),488).
orientation(ms3ggt_0,160,489). coords(ms3ggt_0,76,184,489).  visible(ms3ggt_0,489). happensAt(walking(ms3ggt_0),489). orientation(ms3ggt_1,22,489). coords(ms3ggt_1,255,246,489).  visible(ms3ggt_1,489). happensAt(walking(ms3ggt_1),489).
orientation(ms3ggt_0,160,490). coords(ms3ggt_0,76,184,490).  visible(ms3ggt_0,490). happensAt(walking(ms3ggt_0),490). orientation(ms3ggt_1,22,490). coords(ms3ggt_1,256,248,490).  visible(ms3ggt_1,490). happensAt(walking(ms3ggt_1),490).
orientation(ms3ggt_0,163,491). coords(ms3ggt_0,75,185,491).  visible(ms3ggt_0,491). happensAt(walking(ms3ggt_0),491). orientation(ms3ggt_1,22,491). coords(ms3ggt_1,256,248,491).  visible(ms3ggt_1,491). happensAt(walking(ms3ggt_1),491).
orientation(ms3ggt_0,163,492). coords(ms3ggt_0,75,185,492).  visible(ms3ggt_0,492). happensAt(walking(ms3ggt_0),492). orientation(ms3ggt_1,22,492). coords(ms3ggt_1,256,248,492).  visible(ms3ggt_1,492). happensAt(walking(ms3ggt_1),492).
orientation(ms3ggt_0,163,493). coords(ms3ggt_0,75,186,493).  visible(ms3ggt_0,493). happensAt(walking(ms3ggt_0),493). orientation(ms3ggt_1,22,493). coords(ms3ggt_1,256,248,493).  visible(ms3ggt_1,493). happensAt(walking(ms3ggt_1),493).
orientation(ms3ggt_0,163,494). coords(ms3ggt_0,75,186,494).  visible(ms3ggt_0,494). happensAt(walking(ms3ggt_0),494). orientation(ms3ggt_1,29,494). coords(ms3ggt_1,256,250,494).  visible(ms3ggt_1,494). happensAt(walking(ms3ggt_1),494).
orientation(ms3ggt_0,163,495). coords(ms3ggt_0,75,187,495).  visible(ms3ggt_0,495). happensAt(walking(ms3ggt_0),495). orientation(ms3ggt_1,29,495). coords(ms3ggt_1,259,252,495).  visible(ms3ggt_1,495). happensAt(walking(ms3ggt_1),495).
orientation(ms3ggt_0,163,496). coords(ms3ggt_0,75,188,496).  visible(ms3ggt_0,496). happensAt(walking(ms3ggt_0),496). orientation(ms3ggt_1,29,496). coords(ms3ggt_1,259,253,496).  visible(ms3ggt_1,496). happensAt(walking(ms3ggt_1),496).
orientation(ms3ggt_0,163,497). coords(ms3ggt_0,75,188,497).  visible(ms3ggt_0,497). happensAt(walking(ms3ggt_0),497). orientation(ms3ggt_1,29,497). coords(ms3ggt_1,259,257,497).  visible(ms3ggt_1,497). happensAt(walking(ms3ggt_1),497).
orientation(ms3ggt_0,163,498). coords(ms3ggt_0,75,189,498).  visible(ms3ggt_0,498). happensAt(walking(ms3ggt_0),498). orientation(ms3ggt_1,29,498). coords(ms3ggt_1,260,260,498).  visible(ms3ggt_1,498). happensAt(walking(ms3ggt_1),498).
orientation(ms3ggt_0,163,499). coords(ms3ggt_0,75,189,499).  visible(ms3ggt_0,499). happensAt(walking(ms3ggt_0),499). orientation(ms3ggt_1,29,499). coords(ms3ggt_1,260,263,499).  visible(ms3ggt_1,499). happensAt(walking(ms3ggt_1),499).
orientation(ms3ggt_0,163,500). coords(ms3ggt_0,75,189,500).  visible(ms3ggt_0,500). happensAt(walking(ms3ggt_0),500). orientation(ms3ggt_1,29,500). coords(ms3ggt_1,262,267,500).  visible(ms3ggt_1,500). happensAt(walking(ms3ggt_1),500).
orientation(ms3ggt_0,163,501). coords(ms3ggt_0,75,190,501).  visible(ms3ggt_0,501). happensAt(walking(ms3ggt_0),501). orientation(ms3ggt_1,29,501). coords(ms3ggt_1,263,269,501).  visible(ms3ggt_1,501). happensAt(walking(ms3ggt_1),501).
orientation(ms3ggt_0,163,502). coords(ms3ggt_0,75,190,502).  visible(ms3ggt_0,502). happensAt(walking(ms3ggt_0),502). orientation(ms3ggt_1,29,502). coords(ms3ggt_1,265,271,502).  visible(ms3ggt_1,502). happensAt(walking(ms3ggt_1),502).
orientation(ms3ggt_0,163,503). coords(ms3ggt_0,74,190,503).  visible(ms3ggt_0,503). happensAt(walking(ms3ggt_0),503). orientation(ms3ggt_1,29,503). coords(ms3ggt_1,267,271,503).  visible(ms3ggt_1,503). happensAt(walking(ms3ggt_1),503).
orientation(ms3ggt_0,163,504). coords(ms3ggt_0,74,190,504).  visible(ms3ggt_0,504). happensAt(walking(ms3ggt_0),504). orientation(ms3ggt_1,29,504). coords(ms3ggt_1,268,271,504).  visible(ms3ggt_1,504). happensAt(walking(ms3ggt_1),504).
orientation(ms3ggt_0,163,505). coords(ms3ggt_0,74,190,505).  visible(ms3ggt_0,505). happensAt(walking(ms3ggt_0),505). orientation(ms3ggt_1,29,505). coords(ms3ggt_1,268,271,505).  visible(ms3ggt_1,505). happensAt(walking(ms3ggt_1),505).
orientation(ms3ggt_0,163,506). coords(ms3ggt_0,74,191,506).  visible(ms3ggt_0,506). happensAt(walking(ms3ggt_0),506). orientation(ms3ggt_1,29,506). coords(ms3ggt_1,268,271,506).  visible(ms3ggt_1,506). happensAt(walking(ms3ggt_1),506).
#show holdsAt/2.