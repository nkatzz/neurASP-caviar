
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
orientation(lb2gt_4,154,516). coords(lb2gt_4,86,87,516).  visible(lb2gt_4,516). happensAt(walking(lb2gt_4),516). orientation(lb2gt_6,151,516). coords(lb2gt_6,72,122,516).  visible(lb2gt_6,516). happensAt(inactive(lb2gt_6),516).
orientation(lb2gt_4,154,517). coords(lb2gt_4,88,87,517).  visible(lb2gt_4,517). happensAt(walking(lb2gt_4),517). orientation(lb2gt_6,151,517). coords(lb2gt_6,72,122,517).  visible(lb2gt_6,517). happensAt(inactive(lb2gt_6),517).
orientation(lb2gt_4,154,518). coords(lb2gt_4,89,86,518).  visible(lb2gt_4,518). happensAt(walking(lb2gt_4),518). orientation(lb2gt_6,151,518). coords(lb2gt_6,72,122,518).  visible(lb2gt_6,518). happensAt(inactive(lb2gt_6),518).
orientation(lb2gt_4,154,519). coords(lb2gt_4,89,85,519).  visible(lb2gt_4,519). happensAt(walking(lb2gt_4),519). orientation(lb2gt_6,151,519). coords(lb2gt_6,72,122,519).  visible(lb2gt_6,519). happensAt(inactive(lb2gt_6),519).
orientation(lb2gt_4,154,520). coords(lb2gt_4,89,84,520).  visible(lb2gt_4,520). happensAt(walking(lb2gt_4),520). orientation(lb2gt_6,151,520). coords(lb2gt_6,72,122,520).  visible(lb2gt_6,520). happensAt(inactive(lb2gt_6),520).
orientation(lb2gt_4,154,521). coords(lb2gt_4,90,83,521).  visible(lb2gt_4,521). happensAt(walking(lb2gt_4),521). orientation(lb2gt_6,151,521). coords(lb2gt_6,72,122,521).  visible(lb2gt_6,521). happensAt(inactive(lb2gt_6),521).
orientation(lb2gt_4,154,522). coords(lb2gt_4,90,82,522).  visible(lb2gt_4,522). happensAt(walking(lb2gt_4),522). orientation(lb2gt_6,151,522). coords(lb2gt_6,72,122,522).  visible(lb2gt_6,522). happensAt(inactive(lb2gt_6),522).
orientation(lb2gt_4,154,523). coords(lb2gt_4,91,81,523).  visible(lb2gt_4,523). happensAt(walking(lb2gt_4),523). orientation(lb2gt_6,151,523). coords(lb2gt_6,72,122,523).  visible(lb2gt_6,523). happensAt(inactive(lb2gt_6),523).
orientation(lb2gt_4,154,524). coords(lb2gt_4,91,79,524).  visible(lb2gt_4,524). happensAt(walking(lb2gt_4),524). orientation(lb2gt_6,151,524). coords(lb2gt_6,72,122,524).  visible(lb2gt_6,524). happensAt(inactive(lb2gt_6),524).
orientation(lb2gt_4,154,525). coords(lb2gt_4,92,77,525).  visible(lb2gt_4,525). happensAt(walking(lb2gt_4),525). orientation(lb2gt_6,151,525). coords(lb2gt_6,72,122,525).  visible(lb2gt_6,525). happensAt(inactive(lb2gt_6),525).
orientation(lb2gt_4,154,526). coords(lb2gt_4,92,76,526).  visible(lb2gt_4,526). happensAt(walking(lb2gt_4),526). orientation(lb2gt_6,151,526). coords(lb2gt_6,72,122,526).  visible(lb2gt_6,526). happensAt(inactive(lb2gt_6),526).
orientation(lb2gt_4,154,527). coords(lb2gt_4,93,74,527).  visible(lb2gt_4,527). happensAt(walking(lb2gt_4),527). orientation(lb2gt_6,151,527). coords(lb2gt_6,72,122,527).  visible(lb2gt_6,527). happensAt(inactive(lb2gt_6),527).
#show holdsAt/2.