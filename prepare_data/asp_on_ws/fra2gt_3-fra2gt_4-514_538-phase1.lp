
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

	orientation(fra2gt_3,139,514). coords(fra2gt_3,44,126,514).  visible(fra2gt_3,514). happensAt(walking(fra2gt_3),514). orientation(fra2gt_4,146,514). coords(fra2gt_4,63,82,514).  visible(fra2gt_4,514). happensAt(walking(fra2gt_4),514).
orientation(fra2gt_3,146,515). coords(fra2gt_3,44,125,515).  visible(fra2gt_3,515). happensAt(walking(fra2gt_3),515). orientation(fra2gt_4,146,515). coords(fra2gt_4,63,82,515).  visible(fra2gt_4,515). happensAt(walking(fra2gt_4),515).
orientation(fra2gt_3,146,516). coords(fra2gt_3,44,125,516).  visible(fra2gt_3,516). happensAt(walking(fra2gt_3),516). orientation(fra2gt_4,146,516). coords(fra2gt_4,63,82,516).  visible(fra2gt_4,516). happensAt(walking(fra2gt_4),516).
orientation(fra2gt_3,146,517). coords(fra2gt_3,44,124,517).  visible(fra2gt_3,517). happensAt(walking(fra2gt_3),517). orientation(fra2gt_4,146,517). coords(fra2gt_4,63,83,517).  visible(fra2gt_4,517). happensAt(walking(fra2gt_4),517).
orientation(fra2gt_3,146,518). coords(fra2gt_3,44,122,518).  visible(fra2gt_3,518). happensAt(walking(fra2gt_3),518). orientation(fra2gt_4,146,518). coords(fra2gt_4,63,84,518).  visible(fra2gt_4,518). happensAt(walking(fra2gt_4),518).
orientation(fra2gt_3,146,519). coords(fra2gt_3,44,122,519).  visible(fra2gt_3,519). happensAt(walking(fra2gt_3),519). orientation(fra2gt_4,146,519). coords(fra2gt_4,63,85,519).  visible(fra2gt_4,519). happensAt(walking(fra2gt_4),519).
orientation(fra2gt_3,146,520). coords(fra2gt_3,44,120,520).  visible(fra2gt_3,520). happensAt(walking(fra2gt_3),520). orientation(fra2gt_4,146,520). coords(fra2gt_4,63,87,520).  visible(fra2gt_4,520). happensAt(walking(fra2gt_4),520).
orientation(fra2gt_3,146,521). coords(fra2gt_3,44,119,521).  visible(fra2gt_3,521). happensAt(walking(fra2gt_3),521). orientation(fra2gt_4,146,521). coords(fra2gt_4,63,87,521).  visible(fra2gt_4,521). happensAt(walking(fra2gt_4),521).
orientation(fra2gt_3,146,522). coords(fra2gt_3,44,119,522).  visible(fra2gt_3,522). happensAt(walking(fra2gt_3),522). orientation(fra2gt_4,146,522). coords(fra2gt_4,63,87,522).  visible(fra2gt_4,522). happensAt(walking(fra2gt_4),522).
orientation(fra2gt_3,146,523). coords(fra2gt_3,44,118,523).  visible(fra2gt_3,523). happensAt(walking(fra2gt_3),523). orientation(fra2gt_4,146,523). coords(fra2gt_4,64,88,523).  visible(fra2gt_4,523). happensAt(walking(fra2gt_4),523).
orientation(fra2gt_3,146,524). coords(fra2gt_3,44,117,524).  visible(fra2gt_3,524). happensAt(walking(fra2gt_3),524). orientation(fra2gt_4,146,524). coords(fra2gt_4,64,88,524).  visible(fra2gt_4,524). happensAt(walking(fra2gt_4),524).
orientation(fra2gt_3,146,525). coords(fra2gt_3,44,116,525).  visible(fra2gt_3,525). happensAt(walking(fra2gt_3),525). orientation(fra2gt_4,146,525). coords(fra2gt_4,64,89,525).  visible(fra2gt_4,525). happensAt(walking(fra2gt_4),525).
orientation(fra2gt_3,146,526). coords(fra2gt_3,44,116,526).  visible(fra2gt_3,526). happensAt(walking(fra2gt_3),526). orientation(fra2gt_4,146,526). coords(fra2gt_4,64,89,526).  visible(fra2gt_4,526). happensAt(walking(fra2gt_4),526).
orientation(fra2gt_3,146,527). coords(fra2gt_3,44,114,527).  visible(fra2gt_3,527). happensAt(walking(fra2gt_3),527). orientation(fra2gt_4,146,527). coords(fra2gt_4,64,89,527).  visible(fra2gt_4,527). happensAt(walking(fra2gt_4),527).
orientation(fra2gt_3,146,528). coords(fra2gt_3,44,114,528).  visible(fra2gt_3,528). happensAt(walking(fra2gt_3),528). orientation(fra2gt_4,146,528). coords(fra2gt_4,64,89,528).  visible(fra2gt_4,528). happensAt(walking(fra2gt_4),528).
orientation(fra2gt_3,146,529). coords(fra2gt_3,44,113,529).  visible(fra2gt_3,529). happensAt(walking(fra2gt_3),529). orientation(fra2gt_4,146,529). coords(fra2gt_4,65,90,529).  visible(fra2gt_4,529). happensAt(walking(fra2gt_4),529).
orientation(fra2gt_3,146,530). coords(fra2gt_3,44,113,530).  visible(fra2gt_3,530). happensAt(walking(fra2gt_3),530). orientation(fra2gt_4,146,530). coords(fra2gt_4,65,89,530).  visible(fra2gt_4,530). happensAt(walking(fra2gt_4),530).
orientation(fra2gt_3,146,531). coords(fra2gt_3,44,108,531).  visible(fra2gt_3,531). happensAt(walking(fra2gt_3),531). orientation(fra2gt_4,146,531). coords(fra2gt_4,65,89,531).  visible(fra2gt_4,531). happensAt(walking(fra2gt_4),531).
orientation(fra2gt_3,146,532). coords(fra2gt_3,44,109,532).  visible(fra2gt_3,532). happensAt(walking(fra2gt_3),532). orientation(fra2gt_4,146,532). coords(fra2gt_4,65,90,532).  visible(fra2gt_4,532). happensAt(walking(fra2gt_4),532).
orientation(fra2gt_3,146,533). coords(fra2gt_3,44,109,533).  visible(fra2gt_3,533). happensAt(walking(fra2gt_3),533). orientation(fra2gt_4,146,533). coords(fra2gt_4,65,90,533).  visible(fra2gt_4,533). happensAt(walking(fra2gt_4),533).
orientation(fra2gt_3,146,534). coords(fra2gt_3,44,107,534).  visible(fra2gt_3,534). happensAt(walking(fra2gt_3),534). orientation(fra2gt_4,146,534). coords(fra2gt_4,65,90,534).  visible(fra2gt_4,534). happensAt(walking(fra2gt_4),534).
orientation(fra2gt_3,146,535). coords(fra2gt_3,45,106,535).  visible(fra2gt_3,535). happensAt(walking(fra2gt_3),535). orientation(fra2gt_4,146,535). coords(fra2gt_4,66,91,535).  visible(fra2gt_4,535). happensAt(walking(fra2gt_4),535).
orientation(fra2gt_3,146,536). coords(fra2gt_3,46,106,536).  visible(fra2gt_3,536). happensAt(walking(fra2gt_3),536). orientation(fra2gt_4,146,536). coords(fra2gt_4,66,91,536).  visible(fra2gt_4,536). happensAt(walking(fra2gt_4),536).
orientation(fra2gt_3,146,537). coords(fra2gt_3,46,104,537).  visible(fra2gt_3,537). happensAt(walking(fra2gt_3),537). orientation(fra2gt_4,146,537). coords(fra2gt_4,67,91,537).  visible(fra2gt_4,537). happensAt(walking(fra2gt_4),537).
orientation(fra2gt_3,146,538). coords(fra2gt_3,46,103,538).  visible(fra2gt_3,538). happensAt(walking(fra2gt_3),538). orientation(fra2gt_4,146,538). coords(fra2gt_4,67,91,538).  visible(fra2gt_4,538). happensAt(walking(fra2gt_4),538).
#show holdsAt/2.