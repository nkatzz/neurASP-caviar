
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

	orientation(mwt2gt_1,153,517). coords(mwt2gt_1,50,155,517).  visible(mwt2gt_1,517). happensAt(walking(mwt2gt_1),517). orientation(mwt2gt_2,152,517). coords(mwt2gt_2,44,166,517).  visible(mwt2gt_2,517). happensAt(walking(mwt2gt_2),517).
orientation(mwt2gt_1,153,518). coords(mwt2gt_1,49,156,518).  visible(mwt2gt_1,518). happensAt(walking(mwt2gt_1),518). orientation(mwt2gt_2,152,518). coords(mwt2gt_2,43,167,518).  visible(mwt2gt_2,518). happensAt(walking(mwt2gt_2),518).
orientation(mwt2gt_1,153,519). coords(mwt2gt_1,48,156,519).  visible(mwt2gt_1,519). happensAt(walking(mwt2gt_1),519). orientation(mwt2gt_2,152,519). coords(mwt2gt_2,43,167,519).  visible(mwt2gt_2,519). happensAt(walking(mwt2gt_2),519).
orientation(mwt2gt_1,153,520). coords(mwt2gt_1,48,156,520).  visible(mwt2gt_1,520). happensAt(walking(mwt2gt_1),520). orientation(mwt2gt_2,152,520). coords(mwt2gt_2,43,167,520).  visible(mwt2gt_2,520). happensAt(walking(mwt2gt_2),520).
orientation(mwt2gt_1,153,521). coords(mwt2gt_1,46,156,521).  visible(mwt2gt_1,521). happensAt(walking(mwt2gt_1),521). orientation(mwt2gt_2,152,521). coords(mwt2gt_2,43,167,521).  visible(mwt2gt_2,521). happensAt(walking(mwt2gt_2),521).
orientation(mwt2gt_1,153,522). coords(mwt2gt_1,46,156,522).  visible(mwt2gt_1,522). happensAt(walking(mwt2gt_1),522). orientation(mwt2gt_2,152,522). coords(mwt2gt_2,42,167,522).  visible(mwt2gt_2,522). happensAt(walking(mwt2gt_2),522).
orientation(mwt2gt_1,153,523). coords(mwt2gt_1,46,156,523).  visible(mwt2gt_1,523). happensAt(walking(mwt2gt_1),523). orientation(mwt2gt_2,152,523). coords(mwt2gt_2,42,167,523).  visible(mwt2gt_2,523). happensAt(walking(mwt2gt_2),523).
orientation(mwt2gt_1,153,524). coords(mwt2gt_1,46,156,524).  visible(mwt2gt_1,524). happensAt(walking(mwt2gt_1),524). orientation(mwt2gt_2,152,524). coords(mwt2gt_2,42,167,524).  visible(mwt2gt_2,524). happensAt(walking(mwt2gt_2),524).
orientation(mwt2gt_1,153,525). coords(mwt2gt_1,46,156,525).  visible(mwt2gt_1,525). happensAt(walking(mwt2gt_1),525). orientation(mwt2gt_2,152,525). coords(mwt2gt_2,42,167,525).  visible(mwt2gt_2,525). happensAt(walking(mwt2gt_2),525).
orientation(mwt2gt_1,153,526). coords(mwt2gt_1,46,156,526).  visible(mwt2gt_1,526). happensAt(walking(mwt2gt_1),526). orientation(mwt2gt_2,152,526). coords(mwt2gt_2,42,167,526).  visible(mwt2gt_2,526). happensAt(walking(mwt2gt_2),526).
orientation(mwt2gt_1,153,527). coords(mwt2gt_1,46,156,527).  visible(mwt2gt_1,527). happensAt(walking(mwt2gt_1),527). orientation(mwt2gt_2,152,527). coords(mwt2gt_2,40,168,527).  visible(mwt2gt_2,527). happensAt(walking(mwt2gt_2),527).
orientation(mwt2gt_1,153,528). coords(mwt2gt_1,46,156,528).  visible(mwt2gt_1,528). happensAt(walking(mwt2gt_1),528). orientation(mwt2gt_2,152,528). coords(mwt2gt_2,40,168,528).  visible(mwt2gt_2,528). happensAt(walking(mwt2gt_2),528).
orientation(mwt2gt_1,153,529). coords(mwt2gt_1,46,156,529).  visible(mwt2gt_1,529). happensAt(walking(mwt2gt_1),529). orientation(mwt2gt_2,152,529). coords(mwt2gt_2,39,169,529).  visible(mwt2gt_2,529). happensAt(walking(mwt2gt_2),529).
orientation(mwt2gt_1,153,530). coords(mwt2gt_1,45,156,530).  visible(mwt2gt_1,530). happensAt(walking(mwt2gt_1),530). orientation(mwt2gt_2,152,530). coords(mwt2gt_2,38,169,530).  visible(mwt2gt_2,530). happensAt(walking(mwt2gt_2),530).
orientation(mwt2gt_1,153,531). coords(mwt2gt_1,45,156,531).  visible(mwt2gt_1,531). happensAt(walking(mwt2gt_1),531). orientation(mwt2gt_2,152,531). coords(mwt2gt_2,38,169,531).  visible(mwt2gt_2,531). happensAt(walking(mwt2gt_2),531).
orientation(mwt2gt_1,153,532). coords(mwt2gt_1,44,156,532).  visible(mwt2gt_1,532). happensAt(walking(mwt2gt_1),532). orientation(mwt2gt_2,152,532). coords(mwt2gt_2,38,169,532).  visible(mwt2gt_2,532). happensAt(walking(mwt2gt_2),532).
orientation(mwt2gt_1,153,533). coords(mwt2gt_1,44,156,533).  visible(mwt2gt_1,533). happensAt(walking(mwt2gt_1),533). orientation(mwt2gt_2,152,533). coords(mwt2gt_2,38,169,533).  visible(mwt2gt_2,533). happensAt(walking(mwt2gt_2),533).
orientation(mwt2gt_1,153,534). coords(mwt2gt_1,44,156,534).  visible(mwt2gt_1,534). happensAt(walking(mwt2gt_1),534). orientation(mwt2gt_2,152,534). coords(mwt2gt_2,38,169,534).  visible(mwt2gt_2,534). happensAt(walking(mwt2gt_2),534).
orientation(mwt2gt_1,153,535). coords(mwt2gt_1,44,156,535).  visible(mwt2gt_1,535). happensAt(walking(mwt2gt_1),535). orientation(mwt2gt_2,152,535). coords(mwt2gt_2,38,169,535).  visible(mwt2gt_2,535). happensAt(walking(mwt2gt_2),535).
orientation(mwt2gt_1,153,536). coords(mwt2gt_1,44,156,536).  visible(mwt2gt_1,536). happensAt(walking(mwt2gt_1),536). orientation(mwt2gt_2,152,536). coords(mwt2gt_2,38,169,536).  visible(mwt2gt_2,536). happensAt(walking(mwt2gt_2),536).
orientation(mwt2gt_1,153,537). coords(mwt2gt_1,43,156,537).  visible(mwt2gt_1,537). happensAt(walking(mwt2gt_1),537). orientation(mwt2gt_2,152,537). coords(mwt2gt_2,38,169,537).  visible(mwt2gt_2,537). happensAt(walking(mwt2gt_2),537).
orientation(mwt2gt_1,153,538). coords(mwt2gt_1,43,156,538).  visible(mwt2gt_1,538). happensAt(walking(mwt2gt_1),538). orientation(mwt2gt_2,152,538). coords(mwt2gt_2,38,168,538).  visible(mwt2gt_2,538). happensAt(walking(mwt2gt_2),538).
orientation(mwt2gt_1,153,539). coords(mwt2gt_1,43,155,539).  visible(mwt2gt_1,539). happensAt(walking(mwt2gt_1),539). orientation(mwt2gt_2,152,539). coords(mwt2gt_2,37,168,539).  visible(mwt2gt_2,539). happensAt(walking(mwt2gt_2),539).
orientation(mwt2gt_1,153,540). coords(mwt2gt_1,42,155,540).  visible(mwt2gt_1,540). happensAt(walking(mwt2gt_1),540). orientation(mwt2gt_2,152,540). coords(mwt2gt_2,37,168,540).  visible(mwt2gt_2,540). happensAt(walking(mwt2gt_2),540).
orientation(mwt2gt_1,153,541). coords(mwt2gt_1,42,155,541).  visible(mwt2gt_1,541). happensAt(walking(mwt2gt_1),541). orientation(mwt2gt_2,152,541). coords(mwt2gt_2,36,168,541).  visible(mwt2gt_2,541). happensAt(walking(mwt2gt_2),541).
#show holdsAt/2.