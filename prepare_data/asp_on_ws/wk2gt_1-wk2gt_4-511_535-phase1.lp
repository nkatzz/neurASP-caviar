
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

	orientation(wk2gt_1,122,511). coords(wk2gt_1,115,53,511).  visible(wk2gt_1,511). happensAt(inactive(wk2gt_1),511). orientation(wk2gt_4,149,511). coords(wk2gt_4,63,96,511).  visible(wk2gt_4,511). happensAt(walking(wk2gt_4),511).
orientation(wk2gt_1,122,512). coords(wk2gt_1,115,53,512).  visible(wk2gt_1,512). happensAt(inactive(wk2gt_1),512). orientation(wk2gt_4,149,512). coords(wk2gt_4,62,96,512).  visible(wk2gt_4,512). happensAt(walking(wk2gt_4),512).
orientation(wk2gt_1,122,513). coords(wk2gt_1,115,53,513).  visible(wk2gt_1,513). happensAt(inactive(wk2gt_1),513). orientation(wk2gt_4,149,513). coords(wk2gt_4,61,96,513).  visible(wk2gt_4,513). happensAt(walking(wk2gt_4),513).
orientation(wk2gt_1,122,514). coords(wk2gt_1,115,53,514).  visible(wk2gt_1,514). happensAt(inactive(wk2gt_1),514). orientation(wk2gt_4,149,514). coords(wk2gt_4,60,96,514).  visible(wk2gt_4,514). happensAt(walking(wk2gt_4),514).
orientation(wk2gt_1,122,515). coords(wk2gt_1,115,53,515).  visible(wk2gt_1,515). happensAt(inactive(wk2gt_1),515). orientation(wk2gt_4,149,515). coords(wk2gt_4,60,96,515).  visible(wk2gt_4,515). happensAt(walking(wk2gt_4),515).
orientation(wk2gt_1,122,516). coords(wk2gt_1,115,53,516).  visible(wk2gt_1,516). happensAt(inactive(wk2gt_1),516). orientation(wk2gt_4,149,516). coords(wk2gt_4,59,96,516).  visible(wk2gt_4,516). happensAt(walking(wk2gt_4),516).
orientation(wk2gt_1,122,517). coords(wk2gt_1,115,53,517).  visible(wk2gt_1,517). happensAt(inactive(wk2gt_1),517). orientation(wk2gt_4,149,517). coords(wk2gt_4,59,96,517).  visible(wk2gt_4,517). happensAt(walking(wk2gt_4),517).
orientation(wk2gt_1,122,518). coords(wk2gt_1,115,53,518).  visible(wk2gt_1,518). happensAt(inactive(wk2gt_1),518). orientation(wk2gt_4,149,518). coords(wk2gt_4,59,97,518).  visible(wk2gt_4,518). happensAt(walking(wk2gt_4),518).
orientation(wk2gt_1,122,519). coords(wk2gt_1,115,53,519).  visible(wk2gt_1,519). happensAt(inactive(wk2gt_1),519). orientation(wk2gt_4,150,519). coords(wk2gt_4,59,97,519).  visible(wk2gt_4,519). happensAt(walking(wk2gt_4),519).
orientation(wk2gt_1,122,520). coords(wk2gt_1,115,53,520).  visible(wk2gt_1,520). happensAt(inactive(wk2gt_1),520). orientation(wk2gt_4,150,520). coords(wk2gt_4,58,97,520).  visible(wk2gt_4,520). happensAt(walking(wk2gt_4),520).
orientation(wk2gt_1,122,521). coords(wk2gt_1,115,53,521).  visible(wk2gt_1,521). happensAt(inactive(wk2gt_1),521). orientation(wk2gt_4,150,521). coords(wk2gt_4,58,97,521).  visible(wk2gt_4,521). happensAt(walking(wk2gt_4),521).
orientation(wk2gt_1,122,522). coords(wk2gt_1,115,53,522).  visible(wk2gt_1,522). happensAt(inactive(wk2gt_1),522). orientation(wk2gt_4,150,522). coords(wk2gt_4,58,97,522).  visible(wk2gt_4,522). happensAt(walking(wk2gt_4),522).
orientation(wk2gt_1,122,523). coords(wk2gt_1,115,53,523).  visible(wk2gt_1,523). happensAt(inactive(wk2gt_1),523). orientation(wk2gt_4,150,523). coords(wk2gt_4,58,97,523).  visible(wk2gt_4,523). happensAt(walking(wk2gt_4),523).
orientation(wk2gt_1,122,524). coords(wk2gt_1,115,53,524).  visible(wk2gt_1,524). happensAt(inactive(wk2gt_1),524). orientation(wk2gt_4,150,524). coords(wk2gt_4,57,97,524).  visible(wk2gt_4,524). happensAt(walking(wk2gt_4),524).
orientation(wk2gt_1,122,525). coords(wk2gt_1,115,53,525).  visible(wk2gt_1,525). happensAt(inactive(wk2gt_1),525). orientation(wk2gt_4,150,525). coords(wk2gt_4,57,97,525).  visible(wk2gt_4,525). happensAt(walking(wk2gt_4),525).
orientation(wk2gt_1,122,526). coords(wk2gt_1,115,53,526).  visible(wk2gt_1,526). happensAt(inactive(wk2gt_1),526). orientation(wk2gt_4,150,526). coords(wk2gt_4,56,97,526).  visible(wk2gt_4,526). happensAt(walking(wk2gt_4),526).
orientation(wk2gt_1,122,527). coords(wk2gt_1,115,53,527).  visible(wk2gt_1,527). happensAt(inactive(wk2gt_1),527). orientation(wk2gt_4,150,527). coords(wk2gt_4,55,97,527).  visible(wk2gt_4,527). happensAt(walking(wk2gt_4),527).
orientation(wk2gt_1,122,528). coords(wk2gt_1,115,53,528).  visible(wk2gt_1,528). happensAt(inactive(wk2gt_1),528). orientation(wk2gt_4,150,528). coords(wk2gt_4,55,97,528).  visible(wk2gt_4,528). happensAt(walking(wk2gt_4),528).
orientation(wk2gt_1,122,529). coords(wk2gt_1,115,53,529).  visible(wk2gt_1,529). happensAt(inactive(wk2gt_1),529). orientation(wk2gt_4,150,529). coords(wk2gt_4,55,100,529).  visible(wk2gt_4,529). happensAt(walking(wk2gt_4),529).
orientation(wk2gt_1,122,530). coords(wk2gt_1,115,53,530).  visible(wk2gt_1,530). happensAt(inactive(wk2gt_1),530). orientation(wk2gt_4,150,530). coords(wk2gt_4,55,100,530).  visible(wk2gt_4,530). happensAt(walking(wk2gt_4),530).
orientation(wk2gt_1,122,531). coords(wk2gt_1,115,53,531).  visible(wk2gt_1,531). happensAt(inactive(wk2gt_1),531). orientation(wk2gt_4,150,531). coords(wk2gt_4,55,100,531).  visible(wk2gt_4,531). happensAt(walking(wk2gt_4),531).
orientation(wk2gt_1,122,532). coords(wk2gt_1,115,53,532).  visible(wk2gt_1,532). happensAt(inactive(wk2gt_1),532). orientation(wk2gt_4,150,532). coords(wk2gt_4,55,101,532).  visible(wk2gt_4,532). happensAt(walking(wk2gt_4),532).
orientation(wk2gt_1,122,533). coords(wk2gt_1,115,53,533).  visible(wk2gt_1,533). happensAt(inactive(wk2gt_1),533). orientation(wk2gt_4,150,533). coords(wk2gt_4,54,101,533).  visible(wk2gt_4,533). happensAt(walking(wk2gt_4),533).
orientation(wk2gt_1,122,534). coords(wk2gt_1,115,53,534).  visible(wk2gt_1,534). happensAt(inactive(wk2gt_1),534). orientation(wk2gt_4,150,534). coords(wk2gt_4,54,101,534).  visible(wk2gt_4,534). happensAt(walking(wk2gt_4),534).
orientation(wk2gt_1,122,535). coords(wk2gt_1,115,53,535).  visible(wk2gt_1,535). happensAt(inactive(wk2gt_1),535). orientation(wk2gt_4,150,535). coords(wk2gt_4,53,101,535).  visible(wk2gt_4,535). happensAt(walking(wk2gt_4),535).
#show holdsAt/2.