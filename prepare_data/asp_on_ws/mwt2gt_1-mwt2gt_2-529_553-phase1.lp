
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
orientation(mwt2gt_1,153,542). coords(mwt2gt_1,42,155,542).  visible(mwt2gt_1,542). happensAt(walking(mwt2gt_1),542). orientation(mwt2gt_2,152,542). coords(mwt2gt_2,35,168,542).  visible(mwt2gt_2,542). happensAt(walking(mwt2gt_2),542).
orientation(mwt2gt_1,153,543). coords(mwt2gt_1,41,155,543).  visible(mwt2gt_1,543). happensAt(walking(mwt2gt_1),543). orientation(mwt2gt_2,152,543). coords(mwt2gt_2,35,168,543).  visible(mwt2gt_2,543). happensAt(walking(mwt2gt_2),543).
orientation(mwt2gt_1,153,544). coords(mwt2gt_1,40,155,544).  visible(mwt2gt_1,544). happensAt(walking(mwt2gt_1),544). orientation(mwt2gt_2,152,544). coords(mwt2gt_2,35,168,544).  visible(mwt2gt_2,544). happensAt(walking(mwt2gt_2),544).
orientation(mwt2gt_1,153,545). coords(mwt2gt_1,40,155,545).  visible(mwt2gt_1,545). happensAt(walking(mwt2gt_1),545). orientation(mwt2gt_2,152,545). coords(mwt2gt_2,35,168,545).  visible(mwt2gt_2,545). happensAt(walking(mwt2gt_2),545).
orientation(mwt2gt_1,153,546). coords(mwt2gt_1,39,155,546).  visible(mwt2gt_1,546). happensAt(walking(mwt2gt_1),546). orientation(mwt2gt_2,152,546). coords(mwt2gt_2,35,167,546).  visible(mwt2gt_2,546). happensAt(walking(mwt2gt_2),546).
orientation(mwt2gt_1,153,547). coords(mwt2gt_1,39,155,547).  visible(mwt2gt_1,547). happensAt(walking(mwt2gt_1),547). orientation(mwt2gt_2,152,547). coords(mwt2gt_2,35,167,547).  visible(mwt2gt_2,547). happensAt(walking(mwt2gt_2),547).
orientation(mwt2gt_1,153,548). coords(mwt2gt_1,39,155,548).  visible(mwt2gt_1,548). happensAt(walking(mwt2gt_1),548). orientation(mwt2gt_2,152,548). coords(mwt2gt_2,35,167,548).  visible(mwt2gt_2,548). happensAt(walking(mwt2gt_2),548).
orientation(mwt2gt_1,153,549). coords(mwt2gt_1,39,155,549).  visible(mwt2gt_1,549). happensAt(walking(mwt2gt_1),549). orientation(mwt2gt_2,152,549). coords(mwt2gt_2,34,167,549).  visible(mwt2gt_2,549). happensAt(walking(mwt2gt_2),549).
orientation(mwt2gt_1,153,550). coords(mwt2gt_1,38,155,550).  visible(mwt2gt_1,550). happensAt(walking(mwt2gt_1),550). orientation(mwt2gt_2,152,550). coords(mwt2gt_2,34,167,550).  visible(mwt2gt_2,550). happensAt(walking(mwt2gt_2),550).
orientation(mwt2gt_1,153,551). coords(mwt2gt_1,38,155,551).  visible(mwt2gt_1,551). happensAt(walking(mwt2gt_1),551). orientation(mwt2gt_2,152,551). coords(mwt2gt_2,34,167,551).  visible(mwt2gt_2,551). happensAt(walking(mwt2gt_2),551).
orientation(mwt2gt_1,153,552). coords(mwt2gt_1,38,155,552).  visible(mwt2gt_1,552). happensAt(walking(mwt2gt_1),552). orientation(mwt2gt_2,152,552). coords(mwt2gt_2,34,167,552).  visible(mwt2gt_2,552). happensAt(walking(mwt2gt_2),552).
orientation(mwt2gt_1,153,553). coords(mwt2gt_1,38,155,553).  visible(mwt2gt_1,553). happensAt(walking(mwt2gt_1),553). orientation(mwt2gt_2,152,553). coords(mwt2gt_2,33,167,553).  visible(mwt2gt_2,553). happensAt(walking(mwt2gt_2),553).
#show holdsAt/2.