
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
orientation(fra1gt_3,175,538). coords(fra1gt_3,101,204,538).  visible(fra1gt_3,538). happensAt(active(fra1gt_3),538). orientation(fra1gt_7,146,538). coords(fra1gt_7,42,101,538).  visible(fra1gt_7,538). happensAt(walking(fra1gt_7),538).
orientation(fra1gt_3,175,539). coords(fra1gt_3,101,204,539).  visible(fra1gt_3,539). happensAt(active(fra1gt_3),539). orientation(fra1gt_7,146,539). coords(fra1gt_7,42,101,539).  visible(fra1gt_7,539). happensAt(walking(fra1gt_7),539).
orientation(fra1gt_3,175,540). coords(fra1gt_3,101,204,540).  visible(fra1gt_3,540). happensAt(active(fra1gt_3),540). orientation(fra1gt_7,146,540). coords(fra1gt_7,42,101,540).  visible(fra1gt_7,540). happensAt(walking(fra1gt_7),540).
orientation(fra1gt_3,175,541). coords(fra1gt_3,101,204,541).  visible(fra1gt_3,541). happensAt(active(fra1gt_3),541). orientation(fra1gt_7,146,541). coords(fra1gt_7,42,101,541).  visible(fra1gt_7,541). happensAt(walking(fra1gt_7),541).
orientation(fra1gt_3,175,542). coords(fra1gt_3,101,204,542).  visible(fra1gt_3,542). happensAt(active(fra1gt_3),542). orientation(fra1gt_7,146,542). coords(fra1gt_7,42,101,542).  visible(fra1gt_7,542). happensAt(walking(fra1gt_7),542).
orientation(fra1gt_3,175,543). coords(fra1gt_3,101,204,543).  visible(fra1gt_3,543). happensAt(active(fra1gt_3),543). orientation(fra1gt_7,146,543). coords(fra1gt_7,43,101,543).  visible(fra1gt_7,543). happensAt(walking(fra1gt_7),543).
orientation(fra1gt_3,175,544). coords(fra1gt_3,101,204,544).  visible(fra1gt_3,544). happensAt(active(fra1gt_3),544). orientation(fra1gt_7,146,544). coords(fra1gt_7,43,101,544).  visible(fra1gt_7,544). happensAt(walking(fra1gt_7),544).
orientation(fra1gt_3,175,545). coords(fra1gt_3,101,204,545).  visible(fra1gt_3,545). happensAt(active(fra1gt_3),545). orientation(fra1gt_7,146,545). coords(fra1gt_7,43,101,545).  visible(fra1gt_7,545). happensAt(walking(fra1gt_7),545).
orientation(fra1gt_3,175,546). coords(fra1gt_3,101,204,546).  visible(fra1gt_3,546). happensAt(active(fra1gt_3),546). orientation(fra1gt_7,146,546). coords(fra1gt_7,43,101,546).  visible(fra1gt_7,546). happensAt(walking(fra1gt_7),546).
orientation(fra1gt_3,175,547). coords(fra1gt_3,101,204,547).  visible(fra1gt_3,547). happensAt(active(fra1gt_3),547). orientation(fra1gt_7,146,547). coords(fra1gt_7,43,101,547).  visible(fra1gt_7,547). happensAt(walking(fra1gt_7),547).
orientation(fra1gt_3,175,548). coords(fra1gt_3,101,204,548).  visible(fra1gt_3,548). happensAt(active(fra1gt_3),548). orientation(fra1gt_7,146,548). coords(fra1gt_7,43,101,548).  visible(fra1gt_7,548). happensAt(walking(fra1gt_7),548).
orientation(fra1gt_3,175,549). coords(fra1gt_3,101,204,549).  visible(fra1gt_3,549). happensAt(active(fra1gt_3),549). orientation(fra1gt_7,146,549). coords(fra1gt_7,43,101,549).  visible(fra1gt_7,549). happensAt(walking(fra1gt_7),549).
#show holdsAt/2.