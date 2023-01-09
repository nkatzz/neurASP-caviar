
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

	orientation(lbpugt_2,132,527). coords(lbpugt_2,193,44,527).  visible(lbpugt_2,527). happensAt(walking(lbpugt_2),527). orientation(lbpugt_4,135,527). coords(lbpugt_4,219,34,527).  visible(lbpugt_4,527). happensAt(inactive(lbpugt_4),527).
orientation(lbpugt_2,132,528). coords(lbpugt_2,192,44,528).  visible(lbpugt_2,528). happensAt(walking(lbpugt_2),528). orientation(lbpugt_4,135,528). coords(lbpugt_4,219,34,528).  visible(lbpugt_4,528). happensAt(inactive(lbpugt_4),528).
orientation(lbpugt_2,132,529). coords(lbpugt_2,192,44,529).  visible(lbpugt_2,529). happensAt(walking(lbpugt_2),529). orientation(lbpugt_4,135,529). coords(lbpugt_4,219,34,529).  visible(lbpugt_4,529). happensAt(inactive(lbpugt_4),529).
orientation(lbpugt_2,132,530). coords(lbpugt_2,192,44,530).  visible(lbpugt_2,530). happensAt(walking(lbpugt_2),530). orientation(lbpugt_4,135,530). coords(lbpugt_4,219,34,530).  visible(lbpugt_4,530). happensAt(inactive(lbpugt_4),530).
orientation(lbpugt_2,132,531). coords(lbpugt_2,191,44,531).  visible(lbpugt_2,531). happensAt(walking(lbpugt_2),531). orientation(lbpugt_4,135,531). coords(lbpugt_4,219,34,531).  visible(lbpugt_4,531). happensAt(inactive(lbpugt_4),531).
orientation(lbpugt_2,132,532). coords(lbpugt_2,190,44,532).  visible(lbpugt_2,532). happensAt(walking(lbpugt_2),532). orientation(lbpugt_4,135,532). coords(lbpugt_4,219,34,532).  visible(lbpugt_4,532). happensAt(inactive(lbpugt_4),532).
orientation(lbpugt_2,132,533). coords(lbpugt_2,189,44,533).  visible(lbpugt_2,533). happensAt(walking(lbpugt_2),533). orientation(lbpugt_4,135,533). coords(lbpugt_4,219,34,533).  visible(lbpugt_4,533). happensAt(inactive(lbpugt_4),533).
orientation(lbpugt_2,132,534). coords(lbpugt_2,188,44,534).  visible(lbpugt_2,534). happensAt(walking(lbpugt_2),534). orientation(lbpugt_4,135,534). coords(lbpugt_4,219,34,534).  visible(lbpugt_4,534). happensAt(inactive(lbpugt_4),534).
orientation(lbpugt_2,132,535). coords(lbpugt_2,187,44,535).  visible(lbpugt_2,535). happensAt(walking(lbpugt_2),535). orientation(lbpugt_4,135,535). coords(lbpugt_4,219,34,535).  visible(lbpugt_4,535). happensAt(inactive(lbpugt_4),535).
orientation(lbpugt_2,132,536). coords(lbpugt_2,186,44,536).  visible(lbpugt_2,536). happensAt(walking(lbpugt_2),536). orientation(lbpugt_4,135,536). coords(lbpugt_4,219,34,536).  visible(lbpugt_4,536). happensAt(inactive(lbpugt_4),536).
orientation(lbpugt_2,132,537). coords(lbpugt_2,186,45,537).  visible(lbpugt_2,537). happensAt(walking(lbpugt_2),537). orientation(lbpugt_4,135,537). coords(lbpugt_4,219,34,537).  visible(lbpugt_4,537). happensAt(inactive(lbpugt_4),537).
orientation(lbpugt_2,132,538). coords(lbpugt_2,184,45,538).  visible(lbpugt_2,538). happensAt(walking(lbpugt_2),538). orientation(lbpugt_4,135,538). coords(lbpugt_4,219,34,538).  visible(lbpugt_4,538). happensAt(inactive(lbpugt_4),538).
orientation(lbpugt_2,132,539). coords(lbpugt_2,183,47,539).  visible(lbpugt_2,539). happensAt(walking(lbpugt_2),539). orientation(lbpugt_4,135,539). coords(lbpugt_4,219,34,539).  visible(lbpugt_4,539). happensAt(inactive(lbpugt_4),539).
orientation(lbpugt_2,132,540). coords(lbpugt_2,183,47,540).  visible(lbpugt_2,540). happensAt(walking(lbpugt_2),540). orientation(lbpugt_4,135,540). coords(lbpugt_4,219,34,540).  visible(lbpugt_4,540). happensAt(inactive(lbpugt_4),540).
orientation(lbpugt_2,132,541). coords(lbpugt_2,183,48,541).  visible(lbpugt_2,541). happensAt(walking(lbpugt_2),541). orientation(lbpugt_4,135,541). coords(lbpugt_4,219,34,541).  visible(lbpugt_4,541). happensAt(inactive(lbpugt_4),541).
orientation(lbpugt_2,132,542). coords(lbpugt_2,183,48,542).  visible(lbpugt_2,542). happensAt(walking(lbpugt_2),542). orientation(lbpugt_4,135,542). coords(lbpugt_4,219,34,542).  visible(lbpugt_4,542). happensAt(inactive(lbpugt_4),542).
orientation(lbpugt_2,132,543). coords(lbpugt_2,183,48,543).  visible(lbpugt_2,543). happensAt(walking(lbpugt_2),543). orientation(lbpugt_4,135,543). coords(lbpugt_4,219,34,543).  visible(lbpugt_4,543). happensAt(inactive(lbpugt_4),543).
orientation(lbpugt_2,132,544). coords(lbpugt_2,182,48,544).  visible(lbpugt_2,544). happensAt(walking(lbpugt_2),544). orientation(lbpugt_4,135,544). coords(lbpugt_4,219,34,544).  visible(lbpugt_4,544). happensAt(inactive(lbpugt_4),544).
orientation(lbpugt_2,132,545). coords(lbpugt_2,182,48,545).  visible(lbpugt_2,545). happensAt(walking(lbpugt_2),545). orientation(lbpugt_4,135,545). coords(lbpugt_4,219,34,545).  visible(lbpugt_4,545). happensAt(inactive(lbpugt_4),545).
orientation(lbpugt_2,132,546). coords(lbpugt_2,182,48,546).  visible(lbpugt_2,546). happensAt(walking(lbpugt_2),546). orientation(lbpugt_4,135,546). coords(lbpugt_4,219,34,546).  visible(lbpugt_4,546). happensAt(inactive(lbpugt_4),546).
orientation(lbpugt_2,132,547). coords(lbpugt_2,182,48,547).  visible(lbpugt_2,547). happensAt(walking(lbpugt_2),547). orientation(lbpugt_4,135,547). coords(lbpugt_4,219,34,547).  visible(lbpugt_4,547). happensAt(inactive(lbpugt_4),547).
orientation(lbpugt_2,132,548). coords(lbpugt_2,182,49,548).  visible(lbpugt_2,548). happensAt(walking(lbpugt_2),548). orientation(lbpugt_4,135,548). coords(lbpugt_4,219,34,548).  visible(lbpugt_4,548). happensAt(inactive(lbpugt_4),548).
orientation(lbpugt_2,132,549). coords(lbpugt_2,181,49,549).  visible(lbpugt_2,549). happensAt(active(lbpugt_2),549). orientation(lbpugt_4,135,549). coords(lbpugt_4,219,34,549).  visible(lbpugt_4,549). happensAt(inactive(lbpugt_4),549).
orientation(lbpugt_2,132,550). coords(lbpugt_2,181,49,550).  visible(lbpugt_2,550). happensAt(active(lbpugt_2),550). orientation(lbpugt_4,135,550). coords(lbpugt_4,219,34,550).  visible(lbpugt_4,550). happensAt(inactive(lbpugt_4),550).
orientation(lbpugt_2,132,551). coords(lbpugt_2,181,49,551).  visible(lbpugt_2,551). happensAt(active(lbpugt_2),551). orientation(lbpugt_4,135,551). coords(lbpugt_4,219,34,551).  visible(lbpugt_4,551). happensAt(inactive(lbpugt_4),551).
#show holdsAt/2.