
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
orientation(lb2gt_4,154,528). coords(lb2gt_4,94,74,528).  visible(lb2gt_4,528). happensAt(walking(lb2gt_4),528). orientation(lb2gt_6,151,528). coords(lb2gt_6,72,122,528).  visible(lb2gt_6,528). happensAt(inactive(lb2gt_6),528).
orientation(lb2gt_4,154,529). coords(lb2gt_4,96,73,529).  visible(lb2gt_4,529). happensAt(walking(lb2gt_4),529). orientation(lb2gt_6,151,529). coords(lb2gt_6,72,122,529).  visible(lb2gt_6,529). happensAt(inactive(lb2gt_6),529).
orientation(lb2gt_4,154,530). coords(lb2gt_4,97,73,530).  visible(lb2gt_4,530). happensAt(walking(lb2gt_4),530). orientation(lb2gt_6,151,530). coords(lb2gt_6,72,122,530).  visible(lb2gt_6,530). happensAt(inactive(lb2gt_6),530).
orientation(lb2gt_4,154,531). coords(lb2gt_4,99,72,531).  visible(lb2gt_4,531). happensAt(walking(lb2gt_4),531). orientation(lb2gt_6,151,531). coords(lb2gt_6,72,122,531).  visible(lb2gt_6,531). happensAt(inactive(lb2gt_6),531).
orientation(lb2gt_4,154,532). coords(lb2gt_4,100,72,532).  visible(lb2gt_4,532). happensAt(walking(lb2gt_4),532). orientation(lb2gt_6,151,532). coords(lb2gt_6,72,122,532).  visible(lb2gt_6,532). happensAt(inactive(lb2gt_6),532).
orientation(lb2gt_4,154,533). coords(lb2gt_4,100,71,533).  visible(lb2gt_4,533). happensAt(walking(lb2gt_4),533). orientation(lb2gt_6,151,533). coords(lb2gt_6,72,122,533).  visible(lb2gt_6,533). happensAt(inactive(lb2gt_6),533).
orientation(lb2gt_4,154,534). coords(lb2gt_4,102,70,534).  visible(lb2gt_4,534). happensAt(walking(lb2gt_4),534). orientation(lb2gt_6,151,534). coords(lb2gt_6,72,122,534).  visible(lb2gt_6,534). happensAt(inactive(lb2gt_6),534).
orientation(lb2gt_4,154,535). coords(lb2gt_4,102,68,535).  visible(lb2gt_4,535). happensAt(walking(lb2gt_4),535). orientation(lb2gt_6,151,535). coords(lb2gt_6,72,122,535).  visible(lb2gt_6,535). happensAt(inactive(lb2gt_6),535).
orientation(lb2gt_4,154,536). coords(lb2gt_4,103,67,536).  visible(lb2gt_4,536). happensAt(walking(lb2gt_4),536). orientation(lb2gt_6,151,536). coords(lb2gt_6,72,122,536).  visible(lb2gt_6,536). happensAt(inactive(lb2gt_6),536).
orientation(lb2gt_4,154,537). coords(lb2gt_4,103,66,537).  visible(lb2gt_4,537). happensAt(walking(lb2gt_4),537). orientation(lb2gt_6,151,537). coords(lb2gt_6,72,122,537).  visible(lb2gt_6,537). happensAt(inactive(lb2gt_6),537).
orientation(lb2gt_4,154,538). coords(lb2gt_4,103,64,538).  visible(lb2gt_4,538). happensAt(walking(lb2gt_4),538). orientation(lb2gt_6,151,538). coords(lb2gt_6,72,122,538).  visible(lb2gt_6,538). happensAt(inactive(lb2gt_6),538).
orientation(lb2gt_4,154,539). coords(lb2gt_4,103,62,539).  visible(lb2gt_4,539). happensAt(walking(lb2gt_4),539). orientation(lb2gt_6,151,539). coords(lb2gt_6,72,122,539).  visible(lb2gt_6,539). happensAt(inactive(lb2gt_6),539).
#show holdsAt/2.