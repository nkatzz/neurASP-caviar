
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

	orientation(fomdgt3_1,150,506). coords(fomdgt3_1,46,102,506).  visible(fomdgt3_1,506). happensAt(inactive(fomdgt3_1),506). orientation(fomdgt3_7,159,506). coords(fomdgt3_7,129,184,506).  visible(fomdgt3_7,506). happensAt(walking(fomdgt3_7),506).
orientation(fomdgt3_1,150,507). coords(fomdgt3_1,46,102,507).  visible(fomdgt3_1,507). happensAt(inactive(fomdgt3_1),507). orientation(fomdgt3_7,159,507). coords(fomdgt3_7,130,184,507).  visible(fomdgt3_7,507). happensAt(walking(fomdgt3_7),507).
orientation(fomdgt3_1,150,508). coords(fomdgt3_1,46,102,508).  visible(fomdgt3_1,508). happensAt(inactive(fomdgt3_1),508). orientation(fomdgt3_7,159,508). coords(fomdgt3_7,130,184,508).  visible(fomdgt3_7,508). happensAt(walking(fomdgt3_7),508).
orientation(fomdgt3_1,150,509). coords(fomdgt3_1,46,102,509).  visible(fomdgt3_1,509). happensAt(inactive(fomdgt3_1),509). orientation(fomdgt3_7,159,509). coords(fomdgt3_7,130,185,509).  visible(fomdgt3_7,509). happensAt(walking(fomdgt3_7),509).
orientation(fomdgt3_1,150,510). coords(fomdgt3_1,46,102,510).  visible(fomdgt3_1,510). happensAt(inactive(fomdgt3_1),510). orientation(fomdgt3_7,159,510). coords(fomdgt3_7,131,185,510).  visible(fomdgt3_7,510). happensAt(walking(fomdgt3_7),510).
orientation(fomdgt3_1,150,511). coords(fomdgt3_1,46,102,511).  visible(fomdgt3_1,511). happensAt(inactive(fomdgt3_1),511). orientation(fomdgt3_7,159,511). coords(fomdgt3_7,131,186,511).  visible(fomdgt3_7,511). happensAt(walking(fomdgt3_7),511).
orientation(fomdgt3_1,150,512). coords(fomdgt3_1,46,102,512).  visible(fomdgt3_1,512). happensAt(inactive(fomdgt3_1),512). orientation(fomdgt3_7,159,512). coords(fomdgt3_7,131,189,512).  visible(fomdgt3_7,512). happensAt(walking(fomdgt3_7),512).
orientation(fomdgt3_1,150,513). coords(fomdgt3_1,46,102,513).  visible(fomdgt3_1,513). happensAt(inactive(fomdgt3_1),513). orientation(fomdgt3_7,159,513). coords(fomdgt3_7,131,191,513).  visible(fomdgt3_7,513). happensAt(walking(fomdgt3_7),513).
orientation(fomdgt3_1,150,514). coords(fomdgt3_1,46,102,514).  visible(fomdgt3_1,514). happensAt(inactive(fomdgt3_1),514). orientation(fomdgt3_7,159,514). coords(fomdgt3_7,138,191,514).  visible(fomdgt3_7,514). happensAt(walking(fomdgt3_7),514).
orientation(fomdgt3_1,150,515). coords(fomdgt3_1,46,102,515).  visible(fomdgt3_1,515). happensAt(inactive(fomdgt3_1),515). orientation(fomdgt3_7,159,515). coords(fomdgt3_7,138,191,515).  visible(fomdgt3_7,515). happensAt(walking(fomdgt3_7),515).
orientation(fomdgt3_1,150,516). coords(fomdgt3_1,46,102,516).  visible(fomdgt3_1,516). happensAt(inactive(fomdgt3_1),516). orientation(fomdgt3_7,159,516). coords(fomdgt3_7,141,194,516).  visible(fomdgt3_7,516). happensAt(walking(fomdgt3_7),516).
orientation(fomdgt3_1,150,517). coords(fomdgt3_1,46,102,517).  visible(fomdgt3_1,517). happensAt(inactive(fomdgt3_1),517). orientation(fomdgt3_7,159,517). coords(fomdgt3_7,141,195,517).  visible(fomdgt3_7,517). happensAt(walking(fomdgt3_7),517).
orientation(fomdgt3_1,150,518). coords(fomdgt3_1,46,102,518).  visible(fomdgt3_1,518). happensAt(inactive(fomdgt3_1),518). orientation(fomdgt3_7,159,518). coords(fomdgt3_7,145,195,518).  visible(fomdgt3_7,518). happensAt(walking(fomdgt3_7),518).
orientation(fomdgt3_1,150,519). coords(fomdgt3_1,46,102,519).  visible(fomdgt3_1,519). happensAt(inactive(fomdgt3_1),519). orientation(fomdgt3_7,159,519). coords(fomdgt3_7,146,195,519).  visible(fomdgt3_7,519). happensAt(walking(fomdgt3_7),519).
orientation(fomdgt3_1,150,520). coords(fomdgt3_1,46,102,520).  visible(fomdgt3_1,520). happensAt(inactive(fomdgt3_1),520). orientation(fomdgt3_7,159,520). coords(fomdgt3_7,146,195,520).  visible(fomdgt3_7,520). happensAt(walking(fomdgt3_7),520).
orientation(fomdgt3_1,150,521). coords(fomdgt3_1,46,102,521).  visible(fomdgt3_1,521). happensAt(inactive(fomdgt3_1),521). orientation(fomdgt3_7,159,521). coords(fomdgt3_7,147,196,521).  visible(fomdgt3_7,521). happensAt(walking(fomdgt3_7),521).
orientation(fomdgt3_1,150,522). coords(fomdgt3_1,46,102,522).  visible(fomdgt3_1,522). happensAt(inactive(fomdgt3_1),522). orientation(fomdgt3_7,159,522). coords(fomdgt3_7,148,196,522).  visible(fomdgt3_7,522). happensAt(walking(fomdgt3_7),522).
orientation(fomdgt3_1,150,523). coords(fomdgt3_1,46,102,523).  visible(fomdgt3_1,523). happensAt(inactive(fomdgt3_1),523). orientation(fomdgt3_7,159,523). coords(fomdgt3_7,149,196,523).  visible(fomdgt3_7,523). happensAt(walking(fomdgt3_7),523).
orientation(fomdgt3_1,150,524). coords(fomdgt3_1,46,102,524).  visible(fomdgt3_1,524). happensAt(inactive(fomdgt3_1),524). orientation(fomdgt3_7,159,524). coords(fomdgt3_7,149,199,524).  visible(fomdgt3_7,524). happensAt(walking(fomdgt3_7),524).
orientation(fomdgt3_1,150,525). coords(fomdgt3_1,46,102,525).  visible(fomdgt3_1,525). happensAt(inactive(fomdgt3_1),525). orientation(fomdgt3_7,159,525). coords(fomdgt3_7,150,199,525).  visible(fomdgt3_7,525). happensAt(walking(fomdgt3_7),525).
orientation(fomdgt3_1,150,526). coords(fomdgt3_1,46,102,526).  visible(fomdgt3_1,526). happensAt(inactive(fomdgt3_1),526). orientation(fomdgt3_7,159,526). coords(fomdgt3_7,150,199,526).  visible(fomdgt3_7,526). happensAt(walking(fomdgt3_7),526).
orientation(fomdgt3_1,150,527). coords(fomdgt3_1,46,102,527).  visible(fomdgt3_1,527). happensAt(inactive(fomdgt3_1),527). orientation(fomdgt3_7,159,527). coords(fomdgt3_7,150,199,527).  visible(fomdgt3_7,527). happensAt(walking(fomdgt3_7),527).
orientation(fomdgt3_1,150,528). coords(fomdgt3_1,46,102,528).  visible(fomdgt3_1,528). happensAt(inactive(fomdgt3_1),528). orientation(fomdgt3_7,159,528). coords(fomdgt3_7,151,200,528).  visible(fomdgt3_7,528). happensAt(walking(fomdgt3_7),528).
orientation(fomdgt3_1,150,529). coords(fomdgt3_1,46,102,529).  visible(fomdgt3_1,529). happensAt(inactive(fomdgt3_1),529). orientation(fomdgt3_7,159,529). coords(fomdgt3_7,155,201,529).  visible(fomdgt3_7,529). happensAt(walking(fomdgt3_7),529).
orientation(fomdgt3_1,150,530). coords(fomdgt3_1,46,102,530).  visible(fomdgt3_1,530). happensAt(inactive(fomdgt3_1),530). orientation(fomdgt3_7,159,530). coords(fomdgt3_7,158,203,530).  visible(fomdgt3_7,530). happensAt(walking(fomdgt3_7),530).
#show holdsAt/2.