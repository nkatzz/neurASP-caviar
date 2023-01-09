
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

	orientation(wk1gt_1,140,487). coords(wk1gt_1,81,72,487).  visible(wk1gt_1,487). happensAt(walking(wk1gt_1),487). orientation(wk1gt_4,153,487). coords(wk1gt_4,56,168,487).  visible(wk1gt_4,487). happensAt(active(wk1gt_4),487).
orientation(wk1gt_1,140,488). coords(wk1gt_1,81,72,488).  visible(wk1gt_1,488). happensAt(walking(wk1gt_1),488). orientation(wk1gt_4,154,488). coords(wk1gt_4,56,168,488).  visible(wk1gt_4,488). happensAt(active(wk1gt_4),488).
orientation(wk1gt_1,140,489). coords(wk1gt_1,81,72,489).  visible(wk1gt_1,489). happensAt(walking(wk1gt_1),489). orientation(wk1gt_4,154,489). coords(wk1gt_4,56,168,489).  visible(wk1gt_4,489). happensAt(active(wk1gt_4),489).
orientation(wk1gt_1,140,490). coords(wk1gt_1,81,72,490).  visible(wk1gt_1,490). happensAt(walking(wk1gt_1),490). orientation(wk1gt_4,154,490). coords(wk1gt_4,56,168,490).  visible(wk1gt_4,490). happensAt(active(wk1gt_4),490).
orientation(wk1gt_1,140,491). coords(wk1gt_1,81,72,491).  visible(wk1gt_1,491). happensAt(walking(wk1gt_1),491). orientation(wk1gt_4,157,491). coords(wk1gt_4,56,169,491).  visible(wk1gt_4,491). happensAt(active(wk1gt_4),491).
orientation(wk1gt_1,140,492). coords(wk1gt_1,81,71,492).  visible(wk1gt_1,492). happensAt(walking(wk1gt_1),492). orientation(wk1gt_4,157,492). coords(wk1gt_4,56,169,492).  visible(wk1gt_4,492). happensAt(active(wk1gt_4),492).
orientation(wk1gt_1,140,493). coords(wk1gt_1,81,70,493).  visible(wk1gt_1,493). happensAt(walking(wk1gt_1),493). orientation(wk1gt_4,157,493). coords(wk1gt_4,56,169,493).  visible(wk1gt_4,493). happensAt(active(wk1gt_4),493).
orientation(wk1gt_1,140,494). coords(wk1gt_1,81,70,494).  visible(wk1gt_1,494). happensAt(walking(wk1gt_1),494). orientation(wk1gt_4,157,494). coords(wk1gt_4,56,170,494).  visible(wk1gt_4,494). happensAt(active(wk1gt_4),494).
orientation(wk1gt_1,140,495). coords(wk1gt_1,81,69,495).  visible(wk1gt_1,495). happensAt(walking(wk1gt_1),495). orientation(wk1gt_4,157,495). coords(wk1gt_4,56,170,495).  visible(wk1gt_4,495). happensAt(active(wk1gt_4),495).
orientation(wk1gt_1,140,496). coords(wk1gt_1,80,69,496).  visible(wk1gt_1,496). happensAt(walking(wk1gt_1),496). orientation(wk1gt_4,157,496). coords(wk1gt_4,56,170,496).  visible(wk1gt_4,496). happensAt(active(wk1gt_4),496).
orientation(wk1gt_1,140,497). coords(wk1gt_1,80,68,497).  visible(wk1gt_1,497). happensAt(walking(wk1gt_1),497). orientation(wk1gt_4,157,497). coords(wk1gt_4,56,170,497).  visible(wk1gt_4,497). happensAt(active(wk1gt_4),497).
orientation(wk1gt_1,140,498). coords(wk1gt_1,80,67,498).  visible(wk1gt_1,498). happensAt(walking(wk1gt_1),498). orientation(wk1gt_4,157,498). coords(wk1gt_4,56,170,498).  visible(wk1gt_4,498). happensAt(active(wk1gt_4),498).
orientation(wk1gt_1,140,499). coords(wk1gt_1,80,67,499).  visible(wk1gt_1,499). happensAt(walking(wk1gt_1),499). orientation(wk1gt_4,157,499). coords(wk1gt_4,56,170,499).  visible(wk1gt_4,499). happensAt(active(wk1gt_4),499).
orientation(wk1gt_1,140,500). coords(wk1gt_1,80,66,500).  visible(wk1gt_1,500). happensAt(walking(wk1gt_1),500). orientation(wk1gt_4,157,500). coords(wk1gt_4,56,170,500).  visible(wk1gt_4,500). happensAt(active(wk1gt_4),500).
orientation(wk1gt_1,140,501). coords(wk1gt_1,80,66,501).  visible(wk1gt_1,501). happensAt(walking(wk1gt_1),501). orientation(wk1gt_4,157,501). coords(wk1gt_4,56,170,501).  visible(wk1gt_4,501). happensAt(active(wk1gt_4),501).
orientation(wk1gt_1,140,502). coords(wk1gt_1,81,66,502).  visible(wk1gt_1,502). happensAt(walking(wk1gt_1),502). orientation(wk1gt_4,157,502). coords(wk1gt_4,56,170,502).  visible(wk1gt_4,502). happensAt(active(wk1gt_4),502).
orientation(wk1gt_1,140,503). coords(wk1gt_1,81,66,503).  visible(wk1gt_1,503). happensAt(walking(wk1gt_1),503). orientation(wk1gt_4,157,503). coords(wk1gt_4,56,170,503).  visible(wk1gt_4,503). happensAt(active(wk1gt_4),503).
orientation(wk1gt_1,140,504). coords(wk1gt_1,81,66,504).  visible(wk1gt_1,504). happensAt(walking(wk1gt_1),504). orientation(wk1gt_4,157,504). coords(wk1gt_4,56,170,504).  visible(wk1gt_4,504). happensAt(active(wk1gt_4),504).
orientation(wk1gt_1,140,505). coords(wk1gt_1,81,66,505).  visible(wk1gt_1,505). happensAt(walking(wk1gt_1),505). orientation(wk1gt_4,157,505). coords(wk1gt_4,56,170,505).  visible(wk1gt_4,505). happensAt(active(wk1gt_4),505).
orientation(wk1gt_1,140,506). coords(wk1gt_1,81,66,506).  visible(wk1gt_1,506). happensAt(walking(wk1gt_1),506). orientation(wk1gt_4,157,506). coords(wk1gt_4,56,170,506).  visible(wk1gt_4,506). happensAt(active(wk1gt_4),506).
orientation(wk1gt_1,140,507). coords(wk1gt_1,81,66,507).  visible(wk1gt_1,507). happensAt(walking(wk1gt_1),507). orientation(wk1gt_4,157,507). coords(wk1gt_4,56,170,507).  visible(wk1gt_4,507). happensAt(active(wk1gt_4),507).
orientation(wk1gt_1,140,508). coords(wk1gt_1,82,66,508).  visible(wk1gt_1,508). happensAt(walking(wk1gt_1),508). orientation(wk1gt_4,157,508). coords(wk1gt_4,56,170,508).  visible(wk1gt_4,508). happensAt(active(wk1gt_4),508).
orientation(wk1gt_1,140,509). coords(wk1gt_1,81,65,509).  visible(wk1gt_1,509). happensAt(walking(wk1gt_1),509). orientation(wk1gt_4,157,509). coords(wk1gt_4,56,170,509).  visible(wk1gt_4,509). happensAt(active(wk1gt_4),509).
orientation(wk1gt_1,140,510). coords(wk1gt_1,83,64,510).  visible(wk1gt_1,510). happensAt(walking(wk1gt_1),510). orientation(wk1gt_4,157,510). coords(wk1gt_4,56,170,510).  visible(wk1gt_4,510). happensAt(active(wk1gt_4),510).
orientation(wk1gt_1,140,511). coords(wk1gt_1,82,63,511).  happensAt(disappear(wk1gt_1),511). happensAt(walking(wk1gt_1),511). orientation(wk1gt_4,157,511). coords(wk1gt_4,56,170,511).  visible(wk1gt_4,511). happensAt(active(wk1gt_4),511).
#show holdsAt/2.