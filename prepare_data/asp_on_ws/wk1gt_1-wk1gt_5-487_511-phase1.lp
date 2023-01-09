
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

	orientation(wk1gt_1,140,487). coords(wk1gt_1,81,72,487).  visible(wk1gt_1,487). happensAt(walking(wk1gt_1),487). orientation(wk1gt_5,162,487). coords(wk1gt_5,57,152,487).  visible(wk1gt_5,487). happensAt(active(wk1gt_5),487).
orientation(wk1gt_1,140,488). coords(wk1gt_1,81,72,488).  visible(wk1gt_1,488). happensAt(walking(wk1gt_1),488). orientation(wk1gt_5,162,488). coords(wk1gt_5,57,152,488).  visible(wk1gt_5,488). happensAt(active(wk1gt_5),488).
orientation(wk1gt_1,140,489). coords(wk1gt_1,81,72,489).  visible(wk1gt_1,489). happensAt(walking(wk1gt_1),489). orientation(wk1gt_5,162,489). coords(wk1gt_5,57,151,489).  visible(wk1gt_5,489). happensAt(active(wk1gt_5),489).
orientation(wk1gt_1,140,490). coords(wk1gt_1,81,72,490).  visible(wk1gt_1,490). happensAt(walking(wk1gt_1),490). orientation(wk1gt_5,162,490). coords(wk1gt_5,57,150,490).  visible(wk1gt_5,490). happensAt(active(wk1gt_5),490).
orientation(wk1gt_1,140,491). coords(wk1gt_1,81,72,491).  visible(wk1gt_1,491). happensAt(walking(wk1gt_1),491). orientation(wk1gt_5,162,491). coords(wk1gt_5,57,150,491).  visible(wk1gt_5,491). happensAt(active(wk1gt_5),491).
orientation(wk1gt_1,140,492). coords(wk1gt_1,81,71,492).  visible(wk1gt_1,492). happensAt(walking(wk1gt_1),492). orientation(wk1gt_5,162,492). coords(wk1gt_5,57,150,492).  visible(wk1gt_5,492). happensAt(active(wk1gt_5),492).
orientation(wk1gt_1,140,493). coords(wk1gt_1,81,70,493).  visible(wk1gt_1,493). happensAt(walking(wk1gt_1),493). orientation(wk1gt_5,162,493). coords(wk1gt_5,57,150,493).  visible(wk1gt_5,493). happensAt(active(wk1gt_5),493).
orientation(wk1gt_1,140,494). coords(wk1gt_1,81,70,494).  visible(wk1gt_1,494). happensAt(walking(wk1gt_1),494). orientation(wk1gt_5,162,494). coords(wk1gt_5,57,150,494).  visible(wk1gt_5,494). happensAt(active(wk1gt_5),494).
orientation(wk1gt_1,140,495). coords(wk1gt_1,81,69,495).  visible(wk1gt_1,495). happensAt(walking(wk1gt_1),495). orientation(wk1gt_5,162,495). coords(wk1gt_5,57,150,495).  visible(wk1gt_5,495). happensAt(active(wk1gt_5),495).
orientation(wk1gt_1,140,496). coords(wk1gt_1,80,69,496).  visible(wk1gt_1,496). happensAt(walking(wk1gt_1),496). orientation(wk1gt_5,162,496). coords(wk1gt_5,57,150,496).  visible(wk1gt_5,496). happensAt(active(wk1gt_5),496).
orientation(wk1gt_1,140,497). coords(wk1gt_1,80,68,497).  visible(wk1gt_1,497). happensAt(walking(wk1gt_1),497). orientation(wk1gt_5,162,497). coords(wk1gt_5,57,150,497).  visible(wk1gt_5,497). happensAt(active(wk1gt_5),497).
orientation(wk1gt_1,140,498). coords(wk1gt_1,80,67,498).  visible(wk1gt_1,498). happensAt(walking(wk1gt_1),498). orientation(wk1gt_5,162,498). coords(wk1gt_5,57,150,498).  visible(wk1gt_5,498). happensAt(inactive(wk1gt_5),498).
orientation(wk1gt_1,140,499). coords(wk1gt_1,80,67,499).  visible(wk1gt_1,499). happensAt(walking(wk1gt_1),499). orientation(wk1gt_5,162,499). coords(wk1gt_5,57,150,499).  visible(wk1gt_5,499). happensAt(inactive(wk1gt_5),499).
orientation(wk1gt_1,140,500). coords(wk1gt_1,80,66,500).  visible(wk1gt_1,500). happensAt(walking(wk1gt_1),500). orientation(wk1gt_5,162,500). coords(wk1gt_5,57,150,500).  visible(wk1gt_5,500). happensAt(inactive(wk1gt_5),500).
orientation(wk1gt_1,140,501). coords(wk1gt_1,80,66,501).  visible(wk1gt_1,501). happensAt(walking(wk1gt_1),501). orientation(wk1gt_5,162,501). coords(wk1gt_5,57,150,501).  visible(wk1gt_5,501). happensAt(inactive(wk1gt_5),501).
orientation(wk1gt_1,140,502). coords(wk1gt_1,81,66,502).  visible(wk1gt_1,502). happensAt(walking(wk1gt_1),502). orientation(wk1gt_5,162,502). coords(wk1gt_5,57,150,502).  visible(wk1gt_5,502). happensAt(inactive(wk1gt_5),502).
orientation(wk1gt_1,140,503). coords(wk1gt_1,81,66,503).  visible(wk1gt_1,503). happensAt(walking(wk1gt_1),503). orientation(wk1gt_5,162,503). coords(wk1gt_5,57,150,503).  visible(wk1gt_5,503). happensAt(inactive(wk1gt_5),503).
orientation(wk1gt_1,140,504). coords(wk1gt_1,81,66,504).  visible(wk1gt_1,504). happensAt(walking(wk1gt_1),504). orientation(wk1gt_5,162,504). coords(wk1gt_5,57,150,504).  visible(wk1gt_5,504). happensAt(inactive(wk1gt_5),504).
orientation(wk1gt_1,140,505). coords(wk1gt_1,81,66,505).  visible(wk1gt_1,505). happensAt(walking(wk1gt_1),505). orientation(wk1gt_5,162,505). coords(wk1gt_5,57,150,505).  visible(wk1gt_5,505). happensAt(inactive(wk1gt_5),505).
orientation(wk1gt_1,140,506). coords(wk1gt_1,81,66,506).  visible(wk1gt_1,506). happensAt(walking(wk1gt_1),506). orientation(wk1gt_5,162,506). coords(wk1gt_5,57,150,506).  visible(wk1gt_5,506). happensAt(inactive(wk1gt_5),506).
orientation(wk1gt_1,140,507). coords(wk1gt_1,81,66,507).  visible(wk1gt_1,507). happensAt(walking(wk1gt_1),507). orientation(wk1gt_5,162,507). coords(wk1gt_5,57,150,507).  visible(wk1gt_5,507). happensAt(inactive(wk1gt_5),507).
orientation(wk1gt_1,140,508). coords(wk1gt_1,82,66,508).  visible(wk1gt_1,508). happensAt(walking(wk1gt_1),508). orientation(wk1gt_5,162,508). coords(wk1gt_5,57,150,508).  visible(wk1gt_5,508). happensAt(inactive(wk1gt_5),508).
orientation(wk1gt_1,140,509). coords(wk1gt_1,81,65,509).  visible(wk1gt_1,509). happensAt(walking(wk1gt_1),509). orientation(wk1gt_5,162,509). coords(wk1gt_5,57,150,509).  visible(wk1gt_5,509). happensAt(inactive(wk1gt_5),509).
orientation(wk1gt_1,140,510). coords(wk1gt_1,83,64,510).  visible(wk1gt_1,510). happensAt(walking(wk1gt_1),510). orientation(wk1gt_5,162,510). coords(wk1gt_5,57,150,510).  visible(wk1gt_5,510). happensAt(inactive(wk1gt_5),510).
orientation(wk1gt_1,140,511). coords(wk1gt_1,82,63,511).  happensAt(disappear(wk1gt_1),511). happensAt(walking(wk1gt_1),511). orientation(wk1gt_5,162,511). coords(wk1gt_5,57,150,511).  visible(wk1gt_5,511). happensAt(inactive(wk1gt_5),511).
#show holdsAt/2.