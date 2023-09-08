
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

	orientation(wk2gt_2,141,511). coords(wk2gt_2,99,63,511).  visible(wk2gt_2,511). happensAt(inactive(wk2gt_2),511). orientation(wk2gt_4,149,511). coords(wk2gt_4,63,96,511).  visible(wk2gt_4,511). happensAt(walking(wk2gt_4),511).
orientation(wk2gt_2,141,512). coords(wk2gt_2,99,63,512).  visible(wk2gt_2,512). happensAt(inactive(wk2gt_2),512). orientation(wk2gt_4,149,512). coords(wk2gt_4,62,96,512).  visible(wk2gt_4,512). happensAt(walking(wk2gt_4),512).
orientation(wk2gt_2,141,513). coords(wk2gt_2,99,63,513).  visible(wk2gt_2,513). happensAt(inactive(wk2gt_2),513). orientation(wk2gt_4,149,513). coords(wk2gt_4,61,96,513).  visible(wk2gt_4,513). happensAt(walking(wk2gt_4),513).
orientation(wk2gt_2,141,514). coords(wk2gt_2,99,63,514).  visible(wk2gt_2,514). happensAt(inactive(wk2gt_2),514). orientation(wk2gt_4,149,514). coords(wk2gt_4,60,96,514).  visible(wk2gt_4,514). happensAt(walking(wk2gt_4),514).
orientation(wk2gt_2,141,515). coords(wk2gt_2,99,63,515).  visible(wk2gt_2,515). happensAt(inactive(wk2gt_2),515). orientation(wk2gt_4,149,515). coords(wk2gt_4,60,96,515).  visible(wk2gt_4,515). happensAt(walking(wk2gt_4),515).
orientation(wk2gt_2,141,516). coords(wk2gt_2,99,63,516).  visible(wk2gt_2,516). happensAt(inactive(wk2gt_2),516). orientation(wk2gt_4,149,516). coords(wk2gt_4,59,96,516).  visible(wk2gt_4,516). happensAt(walking(wk2gt_4),516).
orientation(wk2gt_2,141,517). coords(wk2gt_2,99,63,517).  visible(wk2gt_2,517). happensAt(inactive(wk2gt_2),517). orientation(wk2gt_4,149,517). coords(wk2gt_4,59,96,517).  visible(wk2gt_4,517). happensAt(walking(wk2gt_4),517).
orientation(wk2gt_2,141,518). coords(wk2gt_2,99,63,518).  visible(wk2gt_2,518). happensAt(inactive(wk2gt_2),518). orientation(wk2gt_4,149,518). coords(wk2gt_4,59,97,518).  visible(wk2gt_4,518). happensAt(walking(wk2gt_4),518).
orientation(wk2gt_2,141,519). coords(wk2gt_2,99,63,519).  visible(wk2gt_2,519). happensAt(inactive(wk2gt_2),519). orientation(wk2gt_4,150,519). coords(wk2gt_4,59,97,519).  visible(wk2gt_4,519). happensAt(walking(wk2gt_4),519).
orientation(wk2gt_2,141,520). coords(wk2gt_2,99,63,520).  visible(wk2gt_2,520). happensAt(inactive(wk2gt_2),520). orientation(wk2gt_4,150,520). coords(wk2gt_4,58,97,520).  visible(wk2gt_4,520). happensAt(walking(wk2gt_4),520).
orientation(wk2gt_2,141,521). coords(wk2gt_2,99,63,521).  visible(wk2gt_2,521). happensAt(inactive(wk2gt_2),521). orientation(wk2gt_4,150,521). coords(wk2gt_4,58,97,521).  visible(wk2gt_4,521). happensAt(walking(wk2gt_4),521).
orientation(wk2gt_2,141,522). coords(wk2gt_2,99,63,522).  visible(wk2gt_2,522). happensAt(inactive(wk2gt_2),522). orientation(wk2gt_4,150,522). coords(wk2gt_4,58,97,522).  visible(wk2gt_4,522). happensAt(walking(wk2gt_4),522).
orientation(wk2gt_2,141,523). coords(wk2gt_2,99,63,523).  visible(wk2gt_2,523). happensAt(inactive(wk2gt_2),523). orientation(wk2gt_4,150,523). coords(wk2gt_4,58,97,523).  visible(wk2gt_4,523). happensAt(walking(wk2gt_4),523).
orientation(wk2gt_2,141,524). coords(wk2gt_2,99,63,524).  visible(wk2gt_2,524). happensAt(inactive(wk2gt_2),524). orientation(wk2gt_4,150,524). coords(wk2gt_4,57,97,524).  visible(wk2gt_4,524). happensAt(walking(wk2gt_4),524).
orientation(wk2gt_2,141,525). coords(wk2gt_2,99,63,525).  visible(wk2gt_2,525). happensAt(inactive(wk2gt_2),525). orientation(wk2gt_4,150,525). coords(wk2gt_4,57,97,525).  visible(wk2gt_4,525). happensAt(walking(wk2gt_4),525).
orientation(wk2gt_2,141,526). coords(wk2gt_2,99,63,526).  visible(wk2gt_2,526). happensAt(inactive(wk2gt_2),526). orientation(wk2gt_4,150,526). coords(wk2gt_4,56,97,526).  visible(wk2gt_4,526). happensAt(walking(wk2gt_4),526).
orientation(wk2gt_2,141,527). coords(wk2gt_2,99,63,527).  visible(wk2gt_2,527). happensAt(inactive(wk2gt_2),527). orientation(wk2gt_4,150,527). coords(wk2gt_4,55,97,527).  visible(wk2gt_4,527). happensAt(walking(wk2gt_4),527).
orientation(wk2gt_2,141,528). coords(wk2gt_2,99,63,528).  visible(wk2gt_2,528). happensAt(inactive(wk2gt_2),528). orientation(wk2gt_4,150,528). coords(wk2gt_4,55,97,528).  visible(wk2gt_4,528). happensAt(walking(wk2gt_4),528).
orientation(wk2gt_2,141,529). coords(wk2gt_2,99,63,529).  visible(wk2gt_2,529). happensAt(inactive(wk2gt_2),529). orientation(wk2gt_4,150,529). coords(wk2gt_4,55,100,529).  visible(wk2gt_4,529). happensAt(walking(wk2gt_4),529).
orientation(wk2gt_2,141,530). coords(wk2gt_2,99,63,530).  visible(wk2gt_2,530). happensAt(inactive(wk2gt_2),530). orientation(wk2gt_4,150,530). coords(wk2gt_4,55,100,530).  visible(wk2gt_4,530). happensAt(walking(wk2gt_4),530).
orientation(wk2gt_2,141,531). coords(wk2gt_2,99,63,531).  visible(wk2gt_2,531). happensAt(inactive(wk2gt_2),531). orientation(wk2gt_4,150,531). coords(wk2gt_4,55,100,531).  visible(wk2gt_4,531). happensAt(walking(wk2gt_4),531).
orientation(wk2gt_2,141,532). coords(wk2gt_2,99,63,532).  visible(wk2gt_2,532). happensAt(inactive(wk2gt_2),532). orientation(wk2gt_4,150,532). coords(wk2gt_4,55,101,532).  visible(wk2gt_4,532). happensAt(walking(wk2gt_4),532).
orientation(wk2gt_2,141,533). coords(wk2gt_2,99,63,533).  visible(wk2gt_2,533). happensAt(inactive(wk2gt_2),533). orientation(wk2gt_4,150,533). coords(wk2gt_4,54,101,533).  visible(wk2gt_4,533). happensAt(walking(wk2gt_4),533).
orientation(wk2gt_2,141,534). coords(wk2gt_2,99,63,534).  visible(wk2gt_2,534). happensAt(inactive(wk2gt_2),534). orientation(wk2gt_4,150,534). coords(wk2gt_4,54,101,534).  visible(wk2gt_4,534). happensAt(walking(wk2gt_4),534).
orientation(wk2gt_2,141,535). coords(wk2gt_2,99,63,535).  visible(wk2gt_2,535). happensAt(inactive(wk2gt_2),535). orientation(wk2gt_4,150,535). coords(wk2gt_4,53,101,535).  visible(wk2gt_4,535). happensAt(walking(wk2gt_4),535).
#show holdsAt/2.