
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

	orientation(wk2gt_2,141,487). coords(wk2gt_2,98,63,487).  visible(wk2gt_2,487). happensAt(inactive(wk2gt_2),487). orientation(wk2gt_4,146,487). coords(wk2gt_4,72,97,487).  visible(wk2gt_4,487). happensAt(walking(wk2gt_4),487).
orientation(wk2gt_2,141,488). coords(wk2gt_2,98,63,488).  visible(wk2gt_2,488). happensAt(inactive(wk2gt_2),488). orientation(wk2gt_4,150,488). coords(wk2gt_4,72,97,488).  visible(wk2gt_4,488). happensAt(walking(wk2gt_4),488).
orientation(wk2gt_2,141,489). coords(wk2gt_2,99,63,489).  visible(wk2gt_2,489). happensAt(inactive(wk2gt_2),489). orientation(wk2gt_4,150,489). coords(wk2gt_4,70,97,489).  visible(wk2gt_4,489). happensAt(walking(wk2gt_4),489).
orientation(wk2gt_2,141,490). coords(wk2gt_2,98,63,490).  visible(wk2gt_2,490). happensAt(inactive(wk2gt_2),490). orientation(wk2gt_4,150,490). coords(wk2gt_4,69,97,490).  visible(wk2gt_4,490). happensAt(walking(wk2gt_4),490).
orientation(wk2gt_2,141,491). coords(wk2gt_2,99,63,491).  visible(wk2gt_2,491). happensAt(inactive(wk2gt_2),491). orientation(wk2gt_4,150,491). coords(wk2gt_4,69,97,491).  visible(wk2gt_4,491). happensAt(walking(wk2gt_4),491).
orientation(wk2gt_2,141,492). coords(wk2gt_2,99,63,492).  visible(wk2gt_2,492). happensAt(inactive(wk2gt_2),492). orientation(wk2gt_4,150,492). coords(wk2gt_4,69,97,492).  visible(wk2gt_4,492). happensAt(walking(wk2gt_4),492).
orientation(wk2gt_2,141,493). coords(wk2gt_2,99,63,493).  visible(wk2gt_2,493). happensAt(inactive(wk2gt_2),493). orientation(wk2gt_4,149,493). coords(wk2gt_4,68,97,493).  visible(wk2gt_4,493). happensAt(walking(wk2gt_4),493).
orientation(wk2gt_2,141,494). coords(wk2gt_2,99,63,494).  visible(wk2gt_2,494). happensAt(inactive(wk2gt_2),494). orientation(wk2gt_4,149,494). coords(wk2gt_4,68,97,494).  visible(wk2gt_4,494). happensAt(walking(wk2gt_4),494).
orientation(wk2gt_2,141,495). coords(wk2gt_2,100,63,495).  visible(wk2gt_2,495). happensAt(inactive(wk2gt_2),495). orientation(wk2gt_4,149,495). coords(wk2gt_4,68,97,495).  visible(wk2gt_4,495). happensAt(walking(wk2gt_4),495).
orientation(wk2gt_2,141,496). coords(wk2gt_2,99,63,496).  visible(wk2gt_2,496). happensAt(inactive(wk2gt_2),496). orientation(wk2gt_4,146,496). coords(wk2gt_4,68,97,496).  visible(wk2gt_4,496). happensAt(walking(wk2gt_4),496).
orientation(wk2gt_2,141,497). coords(wk2gt_2,99,63,497).  visible(wk2gt_2,497). happensAt(inactive(wk2gt_2),497). orientation(wk2gt_4,146,497). coords(wk2gt_4,68,97,497).  visible(wk2gt_4,497). happensAt(walking(wk2gt_4),497).
orientation(wk2gt_2,141,498). coords(wk2gt_2,99,63,498).  visible(wk2gt_2,498). happensAt(inactive(wk2gt_2),498). orientation(wk2gt_4,146,498). coords(wk2gt_4,67,97,498).  visible(wk2gt_4,498). happensAt(walking(wk2gt_4),498).
orientation(wk2gt_2,141,499). coords(wk2gt_2,99,63,499).  visible(wk2gt_2,499). happensAt(inactive(wk2gt_2),499). orientation(wk2gt_4,146,499). coords(wk2gt_4,67,97,499).  visible(wk2gt_4,499). happensAt(walking(wk2gt_4),499).
orientation(wk2gt_2,141,500). coords(wk2gt_2,99,63,500).  visible(wk2gt_2,500). happensAt(inactive(wk2gt_2),500). orientation(wk2gt_4,146,500). coords(wk2gt_4,66,96,500).  visible(wk2gt_4,500). happensAt(walking(wk2gt_4),500).
orientation(wk2gt_2,141,501). coords(wk2gt_2,99,63,501).  visible(wk2gt_2,501). happensAt(inactive(wk2gt_2),501). orientation(wk2gt_4,146,501). coords(wk2gt_4,65,96,501).  visible(wk2gt_4,501). happensAt(walking(wk2gt_4),501).
orientation(wk2gt_2,141,502). coords(wk2gt_2,99,63,502).  visible(wk2gt_2,502). happensAt(inactive(wk2gt_2),502). orientation(wk2gt_4,146,502). coords(wk2gt_4,65,96,502).  visible(wk2gt_4,502). happensAt(walking(wk2gt_4),502).
orientation(wk2gt_2,141,503). coords(wk2gt_2,99,63,503).  visible(wk2gt_2,503). happensAt(inactive(wk2gt_2),503). orientation(wk2gt_4,146,503). coords(wk2gt_4,64,96,503).  visible(wk2gt_4,503). happensAt(walking(wk2gt_4),503).
orientation(wk2gt_2,141,504). coords(wk2gt_2,99,63,504).  visible(wk2gt_2,504). happensAt(inactive(wk2gt_2),504). orientation(wk2gt_4,149,504). coords(wk2gt_4,64,96,504).  visible(wk2gt_4,504). happensAt(walking(wk2gt_4),504).
orientation(wk2gt_2,141,505). coords(wk2gt_2,99,63,505).  visible(wk2gt_2,505). happensAt(inactive(wk2gt_2),505). orientation(wk2gt_4,149,505). coords(wk2gt_4,64,96,505).  visible(wk2gt_4,505). happensAt(walking(wk2gt_4),505).
orientation(wk2gt_2,141,506). coords(wk2gt_2,99,63,506).  visible(wk2gt_2,506). happensAt(inactive(wk2gt_2),506). orientation(wk2gt_4,149,506). coords(wk2gt_4,64,96,506).  visible(wk2gt_4,506). happensAt(walking(wk2gt_4),506).
orientation(wk2gt_2,141,507). coords(wk2gt_2,99,63,507).  visible(wk2gt_2,507). happensAt(inactive(wk2gt_2),507). orientation(wk2gt_4,149,507). coords(wk2gt_4,64,96,507).  visible(wk2gt_4,507). happensAt(walking(wk2gt_4),507).
orientation(wk2gt_2,141,508). coords(wk2gt_2,99,63,508).  visible(wk2gt_2,508). happensAt(inactive(wk2gt_2),508). orientation(wk2gt_4,149,508). coords(wk2gt_4,64,96,508).  visible(wk2gt_4,508). happensAt(walking(wk2gt_4),508).
orientation(wk2gt_2,141,509). coords(wk2gt_2,99,63,509).  visible(wk2gt_2,509). happensAt(inactive(wk2gt_2),509). orientation(wk2gt_4,149,509). coords(wk2gt_4,63,96,509).  visible(wk2gt_4,509). happensAt(walking(wk2gt_4),509).
orientation(wk2gt_2,141,510). coords(wk2gt_2,99,63,510).  visible(wk2gt_2,510). happensAt(inactive(wk2gt_2),510). orientation(wk2gt_4,149,510). coords(wk2gt_4,63,96,510).  visible(wk2gt_4,510). happensAt(walking(wk2gt_4),510).
orientation(wk2gt_2,141,511). coords(wk2gt_2,99,63,511).  visible(wk2gt_2,511). happensAt(inactive(wk2gt_2),511). orientation(wk2gt_4,149,511). coords(wk2gt_4,63,96,511).  visible(wk2gt_4,511). happensAt(walking(wk2gt_4),511).
#show holdsAt/2.