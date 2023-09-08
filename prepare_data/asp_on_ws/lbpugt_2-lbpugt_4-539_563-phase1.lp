
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
orientation(lbpugt_2,132,552). coords(lbpugt_2,180,49,552).  visible(lbpugt_2,552). happensAt(active(lbpugt_2),552). orientation(lbpugt_4,135,552). coords(lbpugt_4,219,34,552).  visible(lbpugt_4,552). happensAt(inactive(lbpugt_4),552).
orientation(lbpugt_2,132,553). coords(lbpugt_2,180,49,553).  visible(lbpugt_2,553). happensAt(active(lbpugt_2),553). orientation(lbpugt_4,135,553). coords(lbpugt_4,219,34,553).  visible(lbpugt_4,553). happensAt(inactive(lbpugt_4),553).
orientation(lbpugt_2,132,554). coords(lbpugt_2,180,49,554).  visible(lbpugt_2,554). happensAt(active(lbpugt_2),554). orientation(lbpugt_4,135,554). coords(lbpugt_4,219,34,554).  visible(lbpugt_4,554). happensAt(inactive(lbpugt_4),554).
orientation(lbpugt_2,132,555). coords(lbpugt_2,180,49,555).  visible(lbpugt_2,555). happensAt(active(lbpugt_2),555). orientation(lbpugt_4,135,555). coords(lbpugt_4,219,34,555).  visible(lbpugt_4,555). happensAt(inactive(lbpugt_4),555).
orientation(lbpugt_2,132,556). coords(lbpugt_2,180,49,556).  visible(lbpugt_2,556). happensAt(active(lbpugt_2),556). orientation(lbpugt_4,135,556). coords(lbpugt_4,219,34,556).  visible(lbpugt_4,556). happensAt(inactive(lbpugt_4),556).
orientation(lbpugt_2,132,557). coords(lbpugt_2,180,49,557).  visible(lbpugt_2,557). happensAt(active(lbpugt_2),557). orientation(lbpugt_4,135,557). coords(lbpugt_4,219,34,557).  visible(lbpugt_4,557). happensAt(inactive(lbpugt_4),557).
orientation(lbpugt_2,132,558). coords(lbpugt_2,180,49,558).  visible(lbpugt_2,558). happensAt(active(lbpugt_2),558). orientation(lbpugt_4,135,558). coords(lbpugt_4,219,34,558).  visible(lbpugt_4,558). happensAt(inactive(lbpugt_4),558).
orientation(lbpugt_2,132,559). coords(lbpugt_2,180,49,559).  visible(lbpugt_2,559). happensAt(active(lbpugt_2),559). orientation(lbpugt_4,135,559). coords(lbpugt_4,219,34,559).  visible(lbpugt_4,559). happensAt(inactive(lbpugt_4),559).
orientation(lbpugt_2,132,560). coords(lbpugt_2,180,49,560).  visible(lbpugt_2,560). happensAt(active(lbpugt_2),560). orientation(lbpugt_4,135,560). coords(lbpugt_4,219,34,560).  visible(lbpugt_4,560). happensAt(inactive(lbpugt_4),560).
orientation(lbpugt_2,132,561). coords(lbpugt_2,180,49,561).  visible(lbpugt_2,561). happensAt(active(lbpugt_2),561). orientation(lbpugt_4,135,561). coords(lbpugt_4,219,34,561).  visible(lbpugt_4,561). happensAt(inactive(lbpugt_4),561).
orientation(lbpugt_2,132,562). coords(lbpugt_2,180,49,562).  visible(lbpugt_2,562). happensAt(active(lbpugt_2),562). orientation(lbpugt_4,135,562). coords(lbpugt_4,219,34,562).  visible(lbpugt_4,562). happensAt(inactive(lbpugt_4),562).
orientation(lbpugt_2,132,563). coords(lbpugt_2,180,49,563).  visible(lbpugt_2,563). happensAt(active(lbpugt_2),563). orientation(lbpugt_4,135,563). coords(lbpugt_4,219,34,563).  visible(lbpugt_4,563). happensAt(inactive(lbpugt_4),563).
#show holdsAt/2.