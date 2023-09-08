
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

	orientation(ms3ggt_0,163,512). coords(ms3ggt_0,74,191,512).  visible(ms3ggt_0,512). happensAt(inactive(ms3ggt_0),512). orientation(ms3ggt_2,147,512). coords(ms3ggt_2,88,116,512).  visible(ms3ggt_2,512). happensAt(walking(ms3ggt_2),512).
orientation(ms3ggt_0,163,513). coords(ms3ggt_0,74,191,513).  visible(ms3ggt_0,513). happensAt(inactive(ms3ggt_0),513). orientation(ms3ggt_2,147,513). coords(ms3ggt_2,88,116,513).  visible(ms3ggt_2,513). happensAt(walking(ms3ggt_2),513).
orientation(ms3ggt_0,163,514). coords(ms3ggt_0,74,191,514).  visible(ms3ggt_0,514). happensAt(inactive(ms3ggt_0),514). orientation(ms3ggt_2,147,514). coords(ms3ggt_2,88,116,514).  visible(ms3ggt_2,514). happensAt(walking(ms3ggt_2),514).
orientation(ms3ggt_0,163,515). coords(ms3ggt_0,74,191,515).  visible(ms3ggt_0,515). happensAt(inactive(ms3ggt_0),515). orientation(ms3ggt_2,147,515). coords(ms3ggt_2,88,115,515).  visible(ms3ggt_2,515). happensAt(walking(ms3ggt_2),515).
orientation(ms3ggt_0,163,516). coords(ms3ggt_0,74,191,516).  visible(ms3ggt_0,516). happensAt(inactive(ms3ggt_0),516). orientation(ms3ggt_2,147,516). coords(ms3ggt_2,88,115,516).  visible(ms3ggt_2,516). happensAt(walking(ms3ggt_2),516).
orientation(ms3ggt_0,163,517). coords(ms3ggt_0,74,191,517).  visible(ms3ggt_0,517). happensAt(inactive(ms3ggt_0),517). orientation(ms3ggt_2,147,517). coords(ms3ggt_2,88,115,517).  visible(ms3ggt_2,517). happensAt(walking(ms3ggt_2),517).
orientation(ms3ggt_0,163,518). coords(ms3ggt_0,74,191,518).  visible(ms3ggt_0,518). happensAt(inactive(ms3ggt_0),518). orientation(ms3ggt_2,147,518). coords(ms3ggt_2,88,114,518).  visible(ms3ggt_2,518). happensAt(walking(ms3ggt_2),518).
orientation(ms3ggt_0,163,519). coords(ms3ggt_0,74,191,519).  visible(ms3ggt_0,519). happensAt(inactive(ms3ggt_0),519). orientation(ms3ggt_2,147,519). coords(ms3ggt_2,88,113,519).  visible(ms3ggt_2,519). happensAt(walking(ms3ggt_2),519).
orientation(ms3ggt_0,163,520). coords(ms3ggt_0,74,191,520).  visible(ms3ggt_0,520). happensAt(inactive(ms3ggt_0),520). orientation(ms3ggt_2,147,520). coords(ms3ggt_2,87,112,520).  visible(ms3ggt_2,520). happensAt(walking(ms3ggt_2),520).
orientation(ms3ggt_0,163,521). coords(ms3ggt_0,74,191,521).  visible(ms3ggt_0,521). happensAt(inactive(ms3ggt_0),521). orientation(ms3ggt_2,141,521). coords(ms3ggt_2,87,112,521).  visible(ms3ggt_2,521). happensAt(walking(ms3ggt_2),521).
orientation(ms3ggt_0,163,522). coords(ms3ggt_0,74,191,522).  visible(ms3ggt_0,522). happensAt(inactive(ms3ggt_0),522). orientation(ms3ggt_2,141,522). coords(ms3ggt_2,86,110,522).  visible(ms3ggt_2,522). happensAt(walking(ms3ggt_2),522).
orientation(ms3ggt_0,163,523). coords(ms3ggt_0,74,191,523).  visible(ms3ggt_0,523). happensAt(inactive(ms3ggt_0),523). orientation(ms3ggt_2,141,523). coords(ms3ggt_2,86,110,523).  visible(ms3ggt_2,523). happensAt(walking(ms3ggt_2),523).
orientation(ms3ggt_0,168,524). coords(ms3ggt_0,73,191,524).  visible(ms3ggt_0,524). happensAt(inactive(ms3ggt_0),524). orientation(ms3ggt_2,141,524). coords(ms3ggt_2,86,109,524).  visible(ms3ggt_2,524). happensAt(walking(ms3ggt_2),524).
orientation(ms3ggt_0,168,525). coords(ms3ggt_0,73,191,525).  visible(ms3ggt_0,525). happensAt(inactive(ms3ggt_0),525). orientation(ms3ggt_2,141,525). coords(ms3ggt_2,86,108,525).  visible(ms3ggt_2,525). happensAt(walking(ms3ggt_2),525).
orientation(ms3ggt_0,168,526). coords(ms3ggt_0,73,191,526).  visible(ms3ggt_0,526). happensAt(inactive(ms3ggt_0),526). orientation(ms3ggt_2,141,526). coords(ms3ggt_2,86,108,526).  visible(ms3ggt_2,526). happensAt(walking(ms3ggt_2),526).
orientation(ms3ggt_0,168,527). coords(ms3ggt_0,73,191,527).  visible(ms3ggt_0,527). happensAt(inactive(ms3ggt_0),527). orientation(ms3ggt_2,141,527). coords(ms3ggt_2,86,106,527).  visible(ms3ggt_2,527). happensAt(walking(ms3ggt_2),527).
orientation(ms3ggt_0,168,528). coords(ms3ggt_0,73,191,528).  visible(ms3ggt_0,528). happensAt(inactive(ms3ggt_0),528). orientation(ms3ggt_2,141,528). coords(ms3ggt_2,86,106,528).  visible(ms3ggt_2,528). happensAt(walking(ms3ggt_2),528).
orientation(ms3ggt_0,168,529). coords(ms3ggt_0,73,191,529).  visible(ms3ggt_0,529). happensAt(inactive(ms3ggt_0),529). orientation(ms3ggt_2,141,529). coords(ms3ggt_2,86,106,529).  visible(ms3ggt_2,529). happensAt(walking(ms3ggt_2),529).
orientation(ms3ggt_0,168,530). coords(ms3ggt_0,73,191,530).  visible(ms3ggt_0,530). happensAt(inactive(ms3ggt_0),530). orientation(ms3ggt_2,141,530). coords(ms3ggt_2,86,107,530).  visible(ms3ggt_2,530). happensAt(walking(ms3ggt_2),530).
orientation(ms3ggt_0,168,531). coords(ms3ggt_0,73,191,531).  visible(ms3ggt_0,531). happensAt(inactive(ms3ggt_0),531). orientation(ms3ggt_2,141,531). coords(ms3ggt_2,86,106,531).  visible(ms3ggt_2,531). happensAt(walking(ms3ggt_2),531).
orientation(ms3ggt_0,168,532). coords(ms3ggt_0,73,191,532).  visible(ms3ggt_0,532). happensAt(inactive(ms3ggt_0),532). orientation(ms3ggt_2,141,532). coords(ms3ggt_2,86,106,532).  visible(ms3ggt_2,532). happensAt(walking(ms3ggt_2),532).
orientation(ms3ggt_0,168,533). coords(ms3ggt_0,73,191,533).  visible(ms3ggt_0,533). happensAt(inactive(ms3ggt_0),533). orientation(ms3ggt_2,141,533). coords(ms3ggt_2,85,104,533).  visible(ms3ggt_2,533). happensAt(walking(ms3ggt_2),533).
orientation(ms3ggt_0,168,534). coords(ms3ggt_0,73,191,534).  visible(ms3ggt_0,534). happensAt(inactive(ms3ggt_0),534). orientation(ms3ggt_2,141,534). coords(ms3ggt_2,85,103,534).  visible(ms3ggt_2,534). happensAt(walking(ms3ggt_2),534).
orientation(ms3ggt_0,168,535). coords(ms3ggt_0,73,191,535).  visible(ms3ggt_0,535). happensAt(inactive(ms3ggt_0),535). orientation(ms3ggt_2,151,535). coords(ms3ggt_2,85,103,535).  visible(ms3ggt_2,535). happensAt(walking(ms3ggt_2),535).
orientation(ms3ggt_0,168,536). coords(ms3ggt_0,73,191,536).  visible(ms3ggt_0,536). happensAt(inactive(ms3ggt_0),536). orientation(ms3ggt_2,151,536). coords(ms3ggt_2,85,103,536).  visible(ms3ggt_2,536). happensAt(walking(ms3ggt_2),536).
#show holdsAt/2.