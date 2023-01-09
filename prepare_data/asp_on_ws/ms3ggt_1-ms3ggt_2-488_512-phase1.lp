
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

	orientation(ms3ggt_1,22,488). coords(ms3ggt_1,255,244,488).  visible(ms3ggt_1,488). happensAt(walking(ms3ggt_1),488). orientation(ms3ggt_2,150,488). coords(ms3ggt_2,94,133,488).  visible(ms3ggt_2,488). happensAt(walking(ms3ggt_2),488).
orientation(ms3ggt_1,22,489). coords(ms3ggt_1,255,246,489).  visible(ms3ggt_1,489). happensAt(walking(ms3ggt_1),489). orientation(ms3ggt_2,150,489). coords(ms3ggt_2,94,132,489).  visible(ms3ggt_2,489). happensAt(walking(ms3ggt_2),489).
orientation(ms3ggt_1,22,490). coords(ms3ggt_1,256,248,490).  visible(ms3ggt_1,490). happensAt(walking(ms3ggt_1),490). orientation(ms3ggt_2,150,490). coords(ms3ggt_2,94,132,490).  visible(ms3ggt_2,490). happensAt(walking(ms3ggt_2),490).
orientation(ms3ggt_1,22,491). coords(ms3ggt_1,256,248,491).  visible(ms3ggt_1,491). happensAt(walking(ms3ggt_1),491). orientation(ms3ggt_2,150,491). coords(ms3ggt_2,94,131,491).  visible(ms3ggt_2,491). happensAt(walking(ms3ggt_2),491).
orientation(ms3ggt_1,22,492). coords(ms3ggt_1,256,248,492).  visible(ms3ggt_1,492). happensAt(walking(ms3ggt_1),492). orientation(ms3ggt_2,143,492). coords(ms3ggt_2,94,131,492).  visible(ms3ggt_2,492). happensAt(walking(ms3ggt_2),492).
orientation(ms3ggt_1,22,493). coords(ms3ggt_1,256,248,493).  visible(ms3ggt_1,493). happensAt(walking(ms3ggt_1),493). orientation(ms3ggt_2,143,493). coords(ms3ggt_2,94,129,493).  visible(ms3ggt_2,493). happensAt(walking(ms3ggt_2),493).
orientation(ms3ggt_1,29,494). coords(ms3ggt_1,256,250,494).  visible(ms3ggt_1,494). happensAt(walking(ms3ggt_1),494). orientation(ms3ggt_2,143,494). coords(ms3ggt_2,93,127,494).  visible(ms3ggt_2,494). happensAt(walking(ms3ggt_2),494).
orientation(ms3ggt_1,29,495). coords(ms3ggt_1,259,252,495).  visible(ms3ggt_1,495). happensAt(walking(ms3ggt_1),495). orientation(ms3ggt_2,143,495). coords(ms3ggt_2,93,126,495).  visible(ms3ggt_2,495). happensAt(walking(ms3ggt_2),495).
orientation(ms3ggt_1,29,496). coords(ms3ggt_1,259,253,496).  visible(ms3ggt_1,496). happensAt(walking(ms3ggt_1),496). orientation(ms3ggt_2,143,496). coords(ms3ggt_2,93,125,496).  visible(ms3ggt_2,496). happensAt(walking(ms3ggt_2),496).
orientation(ms3ggt_1,29,497). coords(ms3ggt_1,259,257,497).  visible(ms3ggt_1,497). happensAt(walking(ms3ggt_1),497). orientation(ms3ggt_2,143,497). coords(ms3ggt_2,93,124,497).  visible(ms3ggt_2,497). happensAt(walking(ms3ggt_2),497).
orientation(ms3ggt_1,29,498). coords(ms3ggt_1,260,260,498).  visible(ms3ggt_1,498). happensAt(walking(ms3ggt_1),498). orientation(ms3ggt_2,143,498). coords(ms3ggt_2,93,123,498).  visible(ms3ggt_2,498). happensAt(walking(ms3ggt_2),498).
orientation(ms3ggt_1,29,499). coords(ms3ggt_1,260,263,499).  visible(ms3ggt_1,499). happensAt(walking(ms3ggt_1),499). orientation(ms3ggt_2,143,499). coords(ms3ggt_2,93,122,499).  visible(ms3ggt_2,499). happensAt(walking(ms3ggt_2),499).
orientation(ms3ggt_1,29,500). coords(ms3ggt_1,262,267,500).  visible(ms3ggt_1,500). happensAt(walking(ms3ggt_1),500). orientation(ms3ggt_2,147,500). coords(ms3ggt_2,93,120,500).  visible(ms3ggt_2,500). happensAt(walking(ms3ggt_2),500).
orientation(ms3ggt_1,29,501). coords(ms3ggt_1,263,269,501).  visible(ms3ggt_1,501). happensAt(walking(ms3ggt_1),501). orientation(ms3ggt_2,147,501). coords(ms3ggt_2,92,120,501).  visible(ms3ggt_2,501). happensAt(walking(ms3ggt_2),501).
orientation(ms3ggt_1,29,502). coords(ms3ggt_1,265,271,502).  visible(ms3ggt_1,502). happensAt(walking(ms3ggt_1),502). orientation(ms3ggt_2,147,502). coords(ms3ggt_2,92,120,502).  visible(ms3ggt_2,502). happensAt(walking(ms3ggt_2),502).
orientation(ms3ggt_1,29,503). coords(ms3ggt_1,267,271,503).  visible(ms3ggt_1,503). happensAt(walking(ms3ggt_1),503). orientation(ms3ggt_2,147,503). coords(ms3ggt_2,92,120,503).  visible(ms3ggt_2,503). happensAt(walking(ms3ggt_2),503).
orientation(ms3ggt_1,29,504). coords(ms3ggt_1,268,271,504).  visible(ms3ggt_1,504). happensAt(walking(ms3ggt_1),504). orientation(ms3ggt_2,147,504). coords(ms3ggt_2,92,119,504).  visible(ms3ggt_2,504). happensAt(walking(ms3ggt_2),504).
orientation(ms3ggt_1,29,505). coords(ms3ggt_1,268,271,505).  visible(ms3ggt_1,505). happensAt(walking(ms3ggt_1),505). orientation(ms3ggt_2,147,505). coords(ms3ggt_2,92,119,505).  visible(ms3ggt_2,505). happensAt(walking(ms3ggt_2),505).
orientation(ms3ggt_1,29,506). coords(ms3ggt_1,268,271,506).  visible(ms3ggt_1,506). happensAt(walking(ms3ggt_1),506). orientation(ms3ggt_2,147,506). coords(ms3ggt_2,91,118,506).  visible(ms3ggt_2,506). happensAt(walking(ms3ggt_2),506).
orientation(ms3ggt_1,45,507). coords(ms3ggt_1,269,271,507).  visible(ms3ggt_1,507). happensAt(walking(ms3ggt_1),507). orientation(ms3ggt_2,147,507). coords(ms3ggt_2,90,117,507).  visible(ms3ggt_2,507). happensAt(walking(ms3ggt_2),507).
orientation(ms3ggt_1,49,508). coords(ms3ggt_1,270,271,508).  visible(ms3ggt_1,508). happensAt(walking(ms3ggt_1),508). orientation(ms3ggt_2,147,508). coords(ms3ggt_2,90,117,508).  visible(ms3ggt_2,508). happensAt(walking(ms3ggt_2),508).
orientation(ms3ggt_1,52,509). coords(ms3ggt_1,271,271,509).  visible(ms3ggt_1,509). happensAt(walking(ms3ggt_1),509). orientation(ms3ggt_2,147,509). coords(ms3ggt_2,90,117,509).  visible(ms3ggt_2,509). happensAt(walking(ms3ggt_2),509).
orientation(ms3ggt_1,52,510). coords(ms3ggt_1,273,272,510).  visible(ms3ggt_1,510). happensAt(walking(ms3ggt_1),510). orientation(ms3ggt_2,147,510). coords(ms3ggt_2,89,117,510).  visible(ms3ggt_2,510). happensAt(walking(ms3ggt_2),510).
orientation(ms3ggt_1,74,511). coords(ms3ggt_1,276,273,511).  visible(ms3ggt_1,511). happensAt(walking(ms3ggt_1),511). orientation(ms3ggt_2,147,511). coords(ms3ggt_2,89,117,511).  visible(ms3ggt_2,511). happensAt(walking(ms3ggt_2),511).
orientation(ms3ggt_1,74,512). coords(ms3ggt_1,277,275,512).  visible(ms3ggt_1,512). happensAt(walking(ms3ggt_1),512). orientation(ms3ggt_2,147,512). coords(ms3ggt_2,88,116,512).  visible(ms3ggt_2,512). happensAt(walking(ms3ggt_2),512).
#show holdsAt/2.