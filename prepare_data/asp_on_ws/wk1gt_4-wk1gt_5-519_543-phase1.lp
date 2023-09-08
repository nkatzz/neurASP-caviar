
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

	orientation(wk1gt_4,157,519). coords(wk1gt_4,56,169,519).  visible(wk1gt_4,519). happensAt(inactive(wk1gt_4),519). orientation(wk1gt_5,162,519). coords(wk1gt_5,57,150,519).  visible(wk1gt_5,519). happensAt(inactive(wk1gt_5),519).
orientation(wk1gt_4,157,520). coords(wk1gt_4,56,169,520).  visible(wk1gt_4,520). happensAt(inactive(wk1gt_4),520). orientation(wk1gt_5,162,520). coords(wk1gt_5,57,150,520).  visible(wk1gt_5,520). happensAt(inactive(wk1gt_5),520).
orientation(wk1gt_4,157,521). coords(wk1gt_4,56,169,521).  visible(wk1gt_4,521). happensAt(inactive(wk1gt_4),521). orientation(wk1gt_5,162,521). coords(wk1gt_5,57,150,521).  visible(wk1gt_5,521). happensAt(inactive(wk1gt_5),521).
orientation(wk1gt_4,157,522). coords(wk1gt_4,56,169,522).  visible(wk1gt_4,522). happensAt(inactive(wk1gt_4),522). orientation(wk1gt_5,162,522). coords(wk1gt_5,57,150,522).  visible(wk1gt_5,522). happensAt(inactive(wk1gt_5),522).
orientation(wk1gt_4,157,523). coords(wk1gt_4,56,169,523).  visible(wk1gt_4,523). happensAt(inactive(wk1gt_4),523). orientation(wk1gt_5,162,523). coords(wk1gt_5,57,150,523).  visible(wk1gt_5,523). happensAt(inactive(wk1gt_5),523).
orientation(wk1gt_4,157,524). coords(wk1gt_4,56,169,524).  visible(wk1gt_4,524). happensAt(inactive(wk1gt_4),524). orientation(wk1gt_5,162,524). coords(wk1gt_5,57,150,524).  visible(wk1gt_5,524). happensAt(inactive(wk1gt_5),524).
orientation(wk1gt_4,157,525). coords(wk1gt_4,56,169,525).  visible(wk1gt_4,525). happensAt(inactive(wk1gt_4),525). orientation(wk1gt_5,162,525). coords(wk1gt_5,57,150,525).  visible(wk1gt_5,525). happensAt(inactive(wk1gt_5),525).
orientation(wk1gt_4,157,526). coords(wk1gt_4,56,169,526).  visible(wk1gt_4,526). happensAt(inactive(wk1gt_4),526). orientation(wk1gt_5,162,526). coords(wk1gt_5,57,150,526).  visible(wk1gt_5,526). happensAt(inactive(wk1gt_5),526).
orientation(wk1gt_4,157,527). coords(wk1gt_4,56,169,527).  visible(wk1gt_4,527). happensAt(inactive(wk1gt_4),527). orientation(wk1gt_5,162,527). coords(wk1gt_5,57,150,527).  visible(wk1gt_5,527). happensAt(inactive(wk1gt_5),527).
orientation(wk1gt_4,157,528). coords(wk1gt_4,56,169,528).  visible(wk1gt_4,528). happensAt(inactive(wk1gt_4),528). orientation(wk1gt_5,162,528). coords(wk1gt_5,57,150,528).  visible(wk1gt_5,528). happensAt(inactive(wk1gt_5),528).
orientation(wk1gt_4,157,529). coords(wk1gt_4,56,169,529).  visible(wk1gt_4,529). happensAt(inactive(wk1gt_4),529). orientation(wk1gt_5,162,529). coords(wk1gt_5,57,150,529).  visible(wk1gt_5,529). happensAt(inactive(wk1gt_5),529).
orientation(wk1gt_4,157,530). coords(wk1gt_4,56,169,530).  visible(wk1gt_4,530). happensAt(inactive(wk1gt_4),530). orientation(wk1gt_5,162,530). coords(wk1gt_5,57,150,530).  visible(wk1gt_5,530). happensAt(inactive(wk1gt_5),530).
orientation(wk1gt_4,157,531). coords(wk1gt_4,56,169,531).  visible(wk1gt_4,531). happensAt(inactive(wk1gt_4),531). orientation(wk1gt_5,162,531). coords(wk1gt_5,57,150,531).  visible(wk1gt_5,531). happensAt(inactive(wk1gt_5),531).
orientation(wk1gt_4,157,532). coords(wk1gt_4,56,169,532).  visible(wk1gt_4,532). happensAt(inactive(wk1gt_4),532). orientation(wk1gt_5,162,532). coords(wk1gt_5,57,150,532).  visible(wk1gt_5,532). happensAt(inactive(wk1gt_5),532).
orientation(wk1gt_4,157,533). coords(wk1gt_4,56,169,533).  visible(wk1gt_4,533). happensAt(inactive(wk1gt_4),533). orientation(wk1gt_5,162,533). coords(wk1gt_5,57,150,533).  visible(wk1gt_5,533). happensAt(inactive(wk1gt_5),533).
orientation(wk1gt_4,157,534). coords(wk1gt_4,56,169,534).  visible(wk1gt_4,534). happensAt(inactive(wk1gt_4),534). orientation(wk1gt_5,162,534). coords(wk1gt_5,57,150,534).  visible(wk1gt_5,534). happensAt(inactive(wk1gt_5),534).
orientation(wk1gt_4,157,535). coords(wk1gt_4,56,169,535).  visible(wk1gt_4,535). happensAt(inactive(wk1gt_4),535). orientation(wk1gt_5,162,535). coords(wk1gt_5,57,150,535).  visible(wk1gt_5,535). happensAt(inactive(wk1gt_5),535).
orientation(wk1gt_4,157,536). coords(wk1gt_4,56,169,536).  visible(wk1gt_4,536). happensAt(inactive(wk1gt_4),536). orientation(wk1gt_5,162,536). coords(wk1gt_5,57,150,536).  visible(wk1gt_5,536). happensAt(inactive(wk1gt_5),536).
orientation(wk1gt_4,157,537). coords(wk1gt_4,56,169,537).  visible(wk1gt_4,537). happensAt(inactive(wk1gt_4),537). orientation(wk1gt_5,162,537). coords(wk1gt_5,57,150,537).  visible(wk1gt_5,537). happensAt(inactive(wk1gt_5),537).
orientation(wk1gt_4,157,538). coords(wk1gt_4,56,169,538).  visible(wk1gt_4,538). happensAt(inactive(wk1gt_4),538). orientation(wk1gt_5,162,538). coords(wk1gt_5,57,150,538).  visible(wk1gt_5,538). happensAt(inactive(wk1gt_5),538).
orientation(wk1gt_4,157,539). coords(wk1gt_4,56,169,539).  visible(wk1gt_4,539). happensAt(inactive(wk1gt_4),539). orientation(wk1gt_5,162,539). coords(wk1gt_5,57,150,539).  visible(wk1gt_5,539). happensAt(inactive(wk1gt_5),539).
orientation(wk1gt_4,157,540). coords(wk1gt_4,56,169,540).  visible(wk1gt_4,540). happensAt(inactive(wk1gt_4),540). orientation(wk1gt_5,162,540). coords(wk1gt_5,57,150,540).  visible(wk1gt_5,540). happensAt(inactive(wk1gt_5),540).
orientation(wk1gt_4,157,541). coords(wk1gt_4,56,169,541).  visible(wk1gt_4,541). happensAt(inactive(wk1gt_4),541). orientation(wk1gt_5,162,541). coords(wk1gt_5,57,150,541).  visible(wk1gt_5,541). happensAt(inactive(wk1gt_5),541).
orientation(wk1gt_4,157,542). coords(wk1gt_4,56,169,542).  visible(wk1gt_4,542). happensAt(inactive(wk1gt_4),542). orientation(wk1gt_5,162,542). coords(wk1gt_5,57,150,542).  visible(wk1gt_5,542). happensAt(inactive(wk1gt_5),542).
orientation(wk1gt_4,157,543). coords(wk1gt_4,56,169,543).  visible(wk1gt_4,543). happensAt(inactive(wk1gt_4),543). orientation(wk1gt_5,162,543). coords(wk1gt_5,57,150,543).  visible(wk1gt_5,543). happensAt(inactive(wk1gt_5),543).
#show holdsAt/2.