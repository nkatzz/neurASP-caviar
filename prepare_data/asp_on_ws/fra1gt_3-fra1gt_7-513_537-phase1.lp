
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

	orientation(fra1gt_3,175,513). coords(fra1gt_3,101,204,513).  visible(fra1gt_3,513). happensAt(active(fra1gt_3),513). orientation(fra1gt_7,146,513). coords(fra1gt_7,43,100,513).  visible(fra1gt_7,513). happensAt(walking(fra1gt_7),513).
orientation(fra1gt_3,175,514). coords(fra1gt_3,101,204,514).  visible(fra1gt_3,514). happensAt(active(fra1gt_3),514). orientation(fra1gt_7,146,514). coords(fra1gt_7,43,100,514).  visible(fra1gt_7,514). happensAt(walking(fra1gt_7),514).
orientation(fra1gt_3,175,515). coords(fra1gt_3,101,204,515).  visible(fra1gt_3,515). happensAt(active(fra1gt_3),515). orientation(fra1gt_7,146,515). coords(fra1gt_7,43,100,515).  visible(fra1gt_7,515). happensAt(walking(fra1gt_7),515).
orientation(fra1gt_3,175,516). coords(fra1gt_3,101,204,516).  visible(fra1gt_3,516). happensAt(active(fra1gt_3),516). orientation(fra1gt_7,146,516). coords(fra1gt_7,42,100,516).  visible(fra1gt_7,516). happensAt(walking(fra1gt_7),516).
orientation(fra1gt_3,175,517). coords(fra1gt_3,101,204,517).  visible(fra1gt_3,517). happensAt(active(fra1gt_3),517). orientation(fra1gt_7,146,517). coords(fra1gt_7,42,100,517).  visible(fra1gt_7,517). happensAt(walking(fra1gt_7),517).
orientation(fra1gt_3,175,518). coords(fra1gt_3,101,204,518).  visible(fra1gt_3,518). happensAt(active(fra1gt_3),518). orientation(fra1gt_7,146,518). coords(fra1gt_7,42,100,518).  visible(fra1gt_7,518). happensAt(walking(fra1gt_7),518).
orientation(fra1gt_3,175,519). coords(fra1gt_3,101,204,519).  visible(fra1gt_3,519). happensAt(active(fra1gt_3),519). orientation(fra1gt_7,146,519). coords(fra1gt_7,42,100,519).  visible(fra1gt_7,519). happensAt(walking(fra1gt_7),519).
orientation(fra1gt_3,175,520). coords(fra1gt_3,101,204,520).  visible(fra1gt_3,520). happensAt(active(fra1gt_3),520). orientation(fra1gt_7,146,520). coords(fra1gt_7,42,101,520).  visible(fra1gt_7,520). happensAt(walking(fra1gt_7),520).
orientation(fra1gt_3,175,521). coords(fra1gt_3,101,204,521).  visible(fra1gt_3,521). happensAt(active(fra1gt_3),521). orientation(fra1gt_7,146,521). coords(fra1gt_7,42,101,521).  visible(fra1gt_7,521). happensAt(walking(fra1gt_7),521).
orientation(fra1gt_3,175,522). coords(fra1gt_3,101,204,522).  visible(fra1gt_3,522). happensAt(active(fra1gt_3),522). orientation(fra1gt_7,146,522). coords(fra1gt_7,42,101,522).  visible(fra1gt_7,522). happensAt(walking(fra1gt_7),522).
orientation(fra1gt_3,175,523). coords(fra1gt_3,101,204,523).  visible(fra1gt_3,523). happensAt(active(fra1gt_3),523). orientation(fra1gt_7,146,523). coords(fra1gt_7,42,101,523).  visible(fra1gt_7,523). happensAt(walking(fra1gt_7),523).
orientation(fra1gt_3,175,524). coords(fra1gt_3,101,204,524).  visible(fra1gt_3,524). happensAt(active(fra1gt_3),524). orientation(fra1gt_7,146,524). coords(fra1gt_7,42,101,524).  visible(fra1gt_7,524). happensAt(walking(fra1gt_7),524).
orientation(fra1gt_3,175,525). coords(fra1gt_3,101,204,525).  visible(fra1gt_3,525). happensAt(active(fra1gt_3),525). orientation(fra1gt_7,146,525). coords(fra1gt_7,42,101,525).  visible(fra1gt_7,525). happensAt(walking(fra1gt_7),525).
orientation(fra1gt_3,175,526). coords(fra1gt_3,101,204,526).  visible(fra1gt_3,526). happensAt(active(fra1gt_3),526). orientation(fra1gt_7,146,526). coords(fra1gt_7,42,101,526).  visible(fra1gt_7,526). happensAt(walking(fra1gt_7),526).
orientation(fra1gt_3,175,527). coords(fra1gt_3,101,204,527).  visible(fra1gt_3,527). happensAt(active(fra1gt_3),527). orientation(fra1gt_7,146,527). coords(fra1gt_7,42,101,527).  visible(fra1gt_7,527). happensAt(walking(fra1gt_7),527).
orientation(fra1gt_3,175,528). coords(fra1gt_3,101,204,528).  visible(fra1gt_3,528). happensAt(active(fra1gt_3),528). orientation(fra1gt_7,146,528). coords(fra1gt_7,42,101,528).  visible(fra1gt_7,528). happensAt(walking(fra1gt_7),528).
orientation(fra1gt_3,175,529). coords(fra1gt_3,101,204,529).  visible(fra1gt_3,529). happensAt(active(fra1gt_3),529). orientation(fra1gt_7,146,529). coords(fra1gt_7,42,101,529).  visible(fra1gt_7,529). happensAt(walking(fra1gt_7),529).
orientation(fra1gt_3,175,530). coords(fra1gt_3,101,204,530).  visible(fra1gt_3,530). happensAt(active(fra1gt_3),530). orientation(fra1gt_7,146,530). coords(fra1gt_7,42,101,530).  visible(fra1gt_7,530). happensAt(walking(fra1gt_7),530).
orientation(fra1gt_3,175,531). coords(fra1gt_3,101,204,531).  visible(fra1gt_3,531). happensAt(active(fra1gt_3),531). orientation(fra1gt_7,146,531). coords(fra1gt_7,42,101,531).  visible(fra1gt_7,531). happensAt(walking(fra1gt_7),531).
orientation(fra1gt_3,175,532). coords(fra1gt_3,101,204,532).  visible(fra1gt_3,532). happensAt(active(fra1gt_3),532). orientation(fra1gt_7,146,532). coords(fra1gt_7,42,101,532).  visible(fra1gt_7,532). happensAt(walking(fra1gt_7),532).
orientation(fra1gt_3,175,533). coords(fra1gt_3,101,204,533).  visible(fra1gt_3,533). happensAt(active(fra1gt_3),533). orientation(fra1gt_7,146,533). coords(fra1gt_7,42,101,533).  visible(fra1gt_7,533). happensAt(walking(fra1gt_7),533).
orientation(fra1gt_3,175,534). coords(fra1gt_3,101,204,534).  visible(fra1gt_3,534). happensAt(active(fra1gt_3),534). orientation(fra1gt_7,146,534). coords(fra1gt_7,42,101,534).  visible(fra1gt_7,534). happensAt(walking(fra1gt_7),534).
orientation(fra1gt_3,175,535). coords(fra1gt_3,101,204,535).  visible(fra1gt_3,535). happensAt(active(fra1gt_3),535). orientation(fra1gt_7,146,535). coords(fra1gt_7,42,101,535).  visible(fra1gt_7,535). happensAt(walking(fra1gt_7),535).
orientation(fra1gt_3,175,536). coords(fra1gt_3,101,204,536).  visible(fra1gt_3,536). happensAt(active(fra1gt_3),536). orientation(fra1gt_7,146,536). coords(fra1gt_7,42,101,536).  visible(fra1gt_7,536). happensAt(walking(fra1gt_7),536).
orientation(fra1gt_3,175,537). coords(fra1gt_3,101,204,537).  visible(fra1gt_3,537). happensAt(active(fra1gt_3),537). orientation(fra1gt_7,146,537). coords(fra1gt_7,42,101,537).  visible(fra1gt_7,537). happensAt(walking(fra1gt_7),537).
#show holdsAt/2.