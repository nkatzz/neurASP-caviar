
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

	orientation(lb2gt_4,154,491). coords(lb2gt_4,72,111,491).  visible(lb2gt_4,491). happensAt(walking(lb2gt_4),491). orientation(lb2gt_6,151,491). coords(lb2gt_6,72,122,491).  visible(lb2gt_6,491). happensAt(inactive(lb2gt_6),491).
orientation(lb2gt_4,154,492). coords(lb2gt_4,73,111,492).  visible(lb2gt_4,492). happensAt(walking(lb2gt_4),492). orientation(lb2gt_6,151,492). coords(lb2gt_6,72,122,492).  visible(lb2gt_6,492). happensAt(inactive(lb2gt_6),492).
orientation(lb2gt_4,154,493). coords(lb2gt_4,73,110,493).  visible(lb2gt_4,493). happensAt(walking(lb2gt_4),493). orientation(lb2gt_6,151,493). coords(lb2gt_6,72,122,493).  visible(lb2gt_6,493). happensAt(inactive(lb2gt_6),493).
orientation(lb2gt_4,154,494). coords(lb2gt_4,73,109,494).  visible(lb2gt_4,494). happensAt(walking(lb2gt_4),494). orientation(lb2gt_6,151,494). coords(lb2gt_6,72,122,494).  visible(lb2gt_6,494). happensAt(inactive(lb2gt_6),494).
orientation(lb2gt_4,154,495). coords(lb2gt_4,73,107,495).  visible(lb2gt_4,495). happensAt(walking(lb2gt_4),495). orientation(lb2gt_6,151,495). coords(lb2gt_6,72,122,495).  visible(lb2gt_6,495). happensAt(inactive(lb2gt_6),495).
orientation(lb2gt_4,154,496). coords(lb2gt_4,73,106,496).  visible(lb2gt_4,496). happensAt(walking(lb2gt_4),496). orientation(lb2gt_6,151,496). coords(lb2gt_6,72,122,496).  visible(lb2gt_6,496). happensAt(inactive(lb2gt_6),496).
orientation(lb2gt_4,154,497). coords(lb2gt_4,73,104,497).  visible(lb2gt_4,497). happensAt(walking(lb2gt_4),497). orientation(lb2gt_6,151,497). coords(lb2gt_6,72,122,497).  visible(lb2gt_6,497). happensAt(inactive(lb2gt_6),497).
orientation(lb2gt_4,154,498). coords(lb2gt_4,73,102,498).  visible(lb2gt_4,498). happensAt(walking(lb2gt_4),498). orientation(lb2gt_6,151,498). coords(lb2gt_6,72,122,498).  visible(lb2gt_6,498). happensAt(inactive(lb2gt_6),498).
orientation(lb2gt_4,154,499). coords(lb2gt_4,74,101,499).  visible(lb2gt_4,499). happensAt(walking(lb2gt_4),499). orientation(lb2gt_6,151,499). coords(lb2gt_6,72,122,499).  visible(lb2gt_6,499). happensAt(inactive(lb2gt_6),499).
orientation(lb2gt_4,154,500). coords(lb2gt_4,75,101,500).  visible(lb2gt_4,500). happensAt(walking(lb2gt_4),500). orientation(lb2gt_6,151,500). coords(lb2gt_6,72,122,500).  visible(lb2gt_6,500). happensAt(inactive(lb2gt_6),500).
orientation(lb2gt_4,154,501). coords(lb2gt_4,76,100,501).  visible(lb2gt_4,501). happensAt(walking(lb2gt_4),501). orientation(lb2gt_6,151,501). coords(lb2gt_6,72,122,501).  visible(lb2gt_6,501). happensAt(inactive(lb2gt_6),501).
orientation(lb2gt_4,154,502). coords(lb2gt_4,78,100,502).  visible(lb2gt_4,502). happensAt(walking(lb2gt_4),502). orientation(lb2gt_6,151,502). coords(lb2gt_6,72,122,502).  visible(lb2gt_6,502). happensAt(inactive(lb2gt_6),502).
orientation(lb2gt_4,154,503). coords(lb2gt_4,79,99,503).  visible(lb2gt_4,503). happensAt(walking(lb2gt_4),503). orientation(lb2gt_6,151,503). coords(lb2gt_6,72,122,503).  visible(lb2gt_6,503). happensAt(inactive(lb2gt_6),503).
orientation(lb2gt_4,154,504). coords(lb2gt_4,80,99,504).  visible(lb2gt_4,504). happensAt(walking(lb2gt_4),504). orientation(lb2gt_6,151,504). coords(lb2gt_6,72,122,504).  visible(lb2gt_6,504). happensAt(inactive(lb2gt_6),504).
orientation(lb2gt_4,154,505). coords(lb2gt_4,81,98,505).  visible(lb2gt_4,505). happensAt(walking(lb2gt_4),505). orientation(lb2gt_6,151,505). coords(lb2gt_6,72,122,505).  visible(lb2gt_6,505). happensAt(inactive(lb2gt_6),505).
orientation(lb2gt_4,154,506). coords(lb2gt_4,82,98,506).  visible(lb2gt_4,506). happensAt(walking(lb2gt_4),506). orientation(lb2gt_6,151,506). coords(lb2gt_6,72,122,506).  visible(lb2gt_6,506). happensAt(inactive(lb2gt_6),506).
orientation(lb2gt_4,154,507). coords(lb2gt_4,82,97,507).  visible(lb2gt_4,507). happensAt(walking(lb2gt_4),507). orientation(lb2gt_6,151,507). coords(lb2gt_6,72,122,507).  visible(lb2gt_6,507). happensAt(inactive(lb2gt_6),507).
orientation(lb2gt_4,154,508). coords(lb2gt_4,82,96,508).  visible(lb2gt_4,508). happensAt(walking(lb2gt_4),508). orientation(lb2gt_6,151,508). coords(lb2gt_6,72,122,508).  visible(lb2gt_6,508). happensAt(inactive(lb2gt_6),508).
orientation(lb2gt_4,154,509). coords(lb2gt_4,82,94,509).  visible(lb2gt_4,509). happensAt(walking(lb2gt_4),509). orientation(lb2gt_6,151,509). coords(lb2gt_6,72,122,509).  visible(lb2gt_6,509). happensAt(inactive(lb2gt_6),509).
orientation(lb2gt_4,154,510). coords(lb2gt_4,82,92,510).  visible(lb2gt_4,510). happensAt(walking(lb2gt_4),510). orientation(lb2gt_6,151,510). coords(lb2gt_6,72,122,510).  visible(lb2gt_6,510). happensAt(inactive(lb2gt_6),510).
orientation(lb2gt_4,154,511). coords(lb2gt_4,83,91,511).  visible(lb2gt_4,511). happensAt(walking(lb2gt_4),511). orientation(lb2gt_6,151,511). coords(lb2gt_6,72,122,511).  visible(lb2gt_6,511). happensAt(inactive(lb2gt_6),511).
orientation(lb2gt_4,154,512). coords(lb2gt_4,83,90,512).  visible(lb2gt_4,512). happensAt(walking(lb2gt_4),512). orientation(lb2gt_6,151,512). coords(lb2gt_6,72,122,512).  visible(lb2gt_6,512). happensAt(inactive(lb2gt_6),512).
orientation(lb2gt_4,154,513). coords(lb2gt_4,83,89,513).  visible(lb2gt_4,513). happensAt(walking(lb2gt_4),513). orientation(lb2gt_6,151,513). coords(lb2gt_6,72,122,513).  visible(lb2gt_6,513). happensAt(inactive(lb2gt_6),513).
orientation(lb2gt_4,154,514). coords(lb2gt_4,84,89,514).  visible(lb2gt_4,514). happensAt(walking(lb2gt_4),514). orientation(lb2gt_6,151,514). coords(lb2gt_6,72,122,514).  visible(lb2gt_6,514). happensAt(inactive(lb2gt_6),514).
orientation(lb2gt_4,154,515). coords(lb2gt_4,85,88,515).  visible(lb2gt_4,515). happensAt(walking(lb2gt_4),515). orientation(lb2gt_6,151,515). coords(lb2gt_6,72,122,515).  visible(lb2gt_6,515). happensAt(inactive(lb2gt_6),515).
#show holdsAt/2.