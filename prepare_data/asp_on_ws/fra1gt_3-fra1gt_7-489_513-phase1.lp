
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

	orientation(fra1gt_3,175,489). coords(fra1gt_3,101,204,489).  visible(fra1gt_3,489). happensAt(active(fra1gt_3),489). orientation(fra1gt_7,146,489). coords(fra1gt_7,47,92,489).  visible(fra1gt_7,489). happensAt(walking(fra1gt_7),489).
orientation(fra1gt_3,175,490). coords(fra1gt_3,101,204,490).  visible(fra1gt_3,490). happensAt(active(fra1gt_3),490). orientation(fra1gt_7,146,490). coords(fra1gt_7,47,94,490).  visible(fra1gt_7,490). happensAt(walking(fra1gt_7),490).
orientation(fra1gt_3,175,491). coords(fra1gt_3,101,204,491).  visible(fra1gt_3,491). happensAt(active(fra1gt_3),491). orientation(fra1gt_7,146,491). coords(fra1gt_7,47,94,491).  visible(fra1gt_7,491). happensAt(walking(fra1gt_7),491).
orientation(fra1gt_3,175,492). coords(fra1gt_3,101,204,492).  visible(fra1gt_3,492). happensAt(active(fra1gt_3),492). orientation(fra1gt_7,146,492). coords(fra1gt_7,47,95,492).  visible(fra1gt_7,492). happensAt(walking(fra1gt_7),492).
orientation(fra1gt_3,175,493). coords(fra1gt_3,101,204,493).  visible(fra1gt_3,493). happensAt(active(fra1gt_3),493). orientation(fra1gt_7,146,493). coords(fra1gt_7,47,95,493).  visible(fra1gt_7,493). happensAt(walking(fra1gt_7),493).
orientation(fra1gt_3,175,494). coords(fra1gt_3,101,204,494).  visible(fra1gt_3,494). happensAt(active(fra1gt_3),494). orientation(fra1gt_7,146,494). coords(fra1gt_7,47,95,494).  visible(fra1gt_7,494). happensAt(walking(fra1gt_7),494).
orientation(fra1gt_3,175,495). coords(fra1gt_3,101,204,495).  visible(fra1gt_3,495). happensAt(active(fra1gt_3),495). orientation(fra1gt_7,146,495). coords(fra1gt_7,47,95,495).  visible(fra1gt_7,495). happensAt(walking(fra1gt_7),495).
orientation(fra1gt_3,175,496). coords(fra1gt_3,101,204,496).  visible(fra1gt_3,496). happensAt(active(fra1gt_3),496). orientation(fra1gt_7,146,496). coords(fra1gt_7,47,95,496).  visible(fra1gt_7,496). happensAt(walking(fra1gt_7),496).
orientation(fra1gt_3,175,497). coords(fra1gt_3,101,204,497).  visible(fra1gt_3,497). happensAt(active(fra1gt_3),497). orientation(fra1gt_7,146,497). coords(fra1gt_7,45,95,497).  visible(fra1gt_7,497). happensAt(walking(fra1gt_7),497).
orientation(fra1gt_3,175,498). coords(fra1gt_3,101,204,498).  visible(fra1gt_3,498). happensAt(active(fra1gt_3),498). orientation(fra1gt_7,146,498). coords(fra1gt_7,45,95,498).  visible(fra1gt_7,498). happensAt(walking(fra1gt_7),498).
orientation(fra1gt_3,175,499). coords(fra1gt_3,101,204,499).  visible(fra1gt_3,499). happensAt(active(fra1gt_3),499). orientation(fra1gt_7,146,499). coords(fra1gt_7,45,96,499).  visible(fra1gt_7,499). happensAt(walking(fra1gt_7),499).
orientation(fra1gt_3,175,500). coords(fra1gt_3,101,204,500).  visible(fra1gt_3,500). happensAt(active(fra1gt_3),500). orientation(fra1gt_7,146,500). coords(fra1gt_7,45,96,500).  visible(fra1gt_7,500). happensAt(walking(fra1gt_7),500).
orientation(fra1gt_3,175,501). coords(fra1gt_3,101,204,501).  visible(fra1gt_3,501). happensAt(active(fra1gt_3),501). orientation(fra1gt_7,146,501). coords(fra1gt_7,45,97,501).  visible(fra1gt_7,501). happensAt(walking(fra1gt_7),501).
orientation(fra1gt_3,175,502). coords(fra1gt_3,101,204,502).  visible(fra1gt_3,502). happensAt(active(fra1gt_3),502). orientation(fra1gt_7,146,502). coords(fra1gt_7,45,98,502).  visible(fra1gt_7,502). happensAt(walking(fra1gt_7),502).
orientation(fra1gt_3,175,503). coords(fra1gt_3,101,204,503).  visible(fra1gt_3,503). happensAt(active(fra1gt_3),503). orientation(fra1gt_7,146,503). coords(fra1gt_7,45,98,503).  visible(fra1gt_7,503). happensAt(walking(fra1gt_7),503).
orientation(fra1gt_3,175,504). coords(fra1gt_3,101,204,504).  visible(fra1gt_3,504). happensAt(active(fra1gt_3),504). orientation(fra1gt_7,146,504). coords(fra1gt_7,45,99,504).  visible(fra1gt_7,504). happensAt(walking(fra1gt_7),504).
orientation(fra1gt_3,175,505). coords(fra1gt_3,101,204,505).  visible(fra1gt_3,505). happensAt(active(fra1gt_3),505). orientation(fra1gt_7,146,505). coords(fra1gt_7,45,99,505).  visible(fra1gt_7,505). happensAt(walking(fra1gt_7),505).
orientation(fra1gt_3,175,506). coords(fra1gt_3,101,204,506).  visible(fra1gt_3,506). happensAt(active(fra1gt_3),506). orientation(fra1gt_7,146,506). coords(fra1gt_7,44,100,506).  visible(fra1gt_7,506). happensAt(walking(fra1gt_7),506).
orientation(fra1gt_3,175,507). coords(fra1gt_3,101,204,507).  visible(fra1gt_3,507). happensAt(active(fra1gt_3),507). orientation(fra1gt_7,146,507). coords(fra1gt_7,44,100,507).  visible(fra1gt_7,507). happensAt(walking(fra1gt_7),507).
orientation(fra1gt_3,175,508). coords(fra1gt_3,101,204,508).  visible(fra1gt_3,508). happensAt(active(fra1gt_3),508). orientation(fra1gt_7,146,508). coords(fra1gt_7,44,100,508).  visible(fra1gt_7,508). happensAt(walking(fra1gt_7),508).
orientation(fra1gt_3,175,509). coords(fra1gt_3,101,204,509).  visible(fra1gt_3,509). happensAt(active(fra1gt_3),509). orientation(fra1gt_7,146,509). coords(fra1gt_7,44,100,509).  visible(fra1gt_7,509). happensAt(walking(fra1gt_7),509).
orientation(fra1gt_3,175,510). coords(fra1gt_3,101,204,510).  visible(fra1gt_3,510). happensAt(active(fra1gt_3),510). orientation(fra1gt_7,146,510). coords(fra1gt_7,44,100,510).  visible(fra1gt_7,510). happensAt(walking(fra1gt_7),510).
orientation(fra1gt_3,175,511). coords(fra1gt_3,101,204,511).  visible(fra1gt_3,511). happensAt(active(fra1gt_3),511). orientation(fra1gt_7,146,511). coords(fra1gt_7,44,100,511).  visible(fra1gt_7,511). happensAt(walking(fra1gt_7),511).
orientation(fra1gt_3,175,512). coords(fra1gt_3,101,204,512).  visible(fra1gt_3,512). happensAt(active(fra1gt_3),512). orientation(fra1gt_7,146,512). coords(fra1gt_7,43,100,512).  visible(fra1gt_7,512). happensAt(walking(fra1gt_7),512).
orientation(fra1gt_3,175,513). coords(fra1gt_3,101,204,513).  visible(fra1gt_3,513). happensAt(active(fra1gt_3),513). orientation(fra1gt_7,146,513). coords(fra1gt_7,43,100,513).  visible(fra1gt_7,513). happensAt(walking(fra1gt_7),513).
#show holdsAt/2.