
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

	orientation(wk2gt_1,122,535). coords(wk2gt_1,115,53,535).  visible(wk2gt_1,535). happensAt(inactive(wk2gt_1),535). orientation(wk2gt_4,150,535). coords(wk2gt_4,53,101,535).  visible(wk2gt_4,535). happensAt(walking(wk2gt_4),535).
orientation(wk2gt_1,122,536). coords(wk2gt_1,115,53,536).  visible(wk2gt_1,536). happensAt(inactive(wk2gt_1),536). orientation(wk2gt_4,150,536). coords(wk2gt_4,53,101,536).  visible(wk2gt_4,536). happensAt(walking(wk2gt_4),536).
orientation(wk2gt_1,122,537). coords(wk2gt_1,115,53,537).  visible(wk2gt_1,537). happensAt(inactive(wk2gt_1),537). orientation(wk2gt_4,150,537). coords(wk2gt_4,50,101,537).  visible(wk2gt_4,537). happensAt(walking(wk2gt_4),537).
orientation(wk2gt_1,122,538). coords(wk2gt_1,115,53,538).  visible(wk2gt_1,538). happensAt(inactive(wk2gt_1),538). orientation(wk2gt_4,150,538). coords(wk2gt_4,50,101,538).  visible(wk2gt_4,538). happensAt(walking(wk2gt_4),538).
orientation(wk2gt_1,122,539). coords(wk2gt_1,115,53,539).  visible(wk2gt_1,539). happensAt(inactive(wk2gt_1),539). orientation(wk2gt_4,150,539). coords(wk2gt_4,50,101,539).  visible(wk2gt_4,539). happensAt(walking(wk2gt_4),539).
orientation(wk2gt_1,122,540). coords(wk2gt_1,115,53,540).  visible(wk2gt_1,540). happensAt(inactive(wk2gt_1),540). orientation(wk2gt_4,150,540). coords(wk2gt_4,50,101,540).  visible(wk2gt_4,540). happensAt(walking(wk2gt_4),540).
orientation(wk2gt_1,122,541). coords(wk2gt_1,115,53,541).  visible(wk2gt_1,541). happensAt(inactive(wk2gt_1),541). orientation(wk2gt_4,150,541). coords(wk2gt_4,50,101,541).  visible(wk2gt_4,541). happensAt(walking(wk2gt_4),541).
orientation(wk2gt_1,122,542). coords(wk2gt_1,115,53,542).  visible(wk2gt_1,542). happensAt(inactive(wk2gt_1),542). orientation(wk2gt_4,150,542). coords(wk2gt_4,49,101,542).  visible(wk2gt_4,542). happensAt(walking(wk2gt_4),542).
orientation(wk2gt_1,122,543). coords(wk2gt_1,115,53,543).  visible(wk2gt_1,543). happensAt(inactive(wk2gt_1),543). orientation(wk2gt_4,150,543). coords(wk2gt_4,49,101,543).  visible(wk2gt_4,543). happensAt(walking(wk2gt_4),543).
orientation(wk2gt_1,122,544). coords(wk2gt_1,115,53,544).  visible(wk2gt_1,544). happensAt(inactive(wk2gt_1),544). orientation(wk2gt_4,149,544). coords(wk2gt_4,49,101,544).  visible(wk2gt_4,544). happensAt(walking(wk2gt_4),544).
orientation(wk2gt_1,122,545). coords(wk2gt_1,115,53,545).  visible(wk2gt_1,545). happensAt(inactive(wk2gt_1),545). orientation(wk2gt_4,149,545). coords(wk2gt_4,49,101,545).  visible(wk2gt_4,545). happensAt(walking(wk2gt_4),545).
orientation(wk2gt_1,122,546). coords(wk2gt_1,115,53,546).  visible(wk2gt_1,546). happensAt(inactive(wk2gt_1),546). orientation(wk2gt_4,149,546). coords(wk2gt_4,49,101,546).  visible(wk2gt_4,546). happensAt(walking(wk2gt_4),546).
orientation(wk2gt_1,122,547). coords(wk2gt_1,115,53,547).  visible(wk2gt_1,547). happensAt(inactive(wk2gt_1),547). orientation(wk2gt_4,149,547). coords(wk2gt_4,49,101,547).  visible(wk2gt_4,547). happensAt(walking(wk2gt_4),547).
orientation(wk2gt_1,122,548). coords(wk2gt_1,115,53,548).  visible(wk2gt_1,548). happensAt(inactive(wk2gt_1),548). orientation(wk2gt_4,149,548). coords(wk2gt_4,49,101,548).  visible(wk2gt_4,548). happensAt(walking(wk2gt_4),548).
orientation(wk2gt_1,122,549). coords(wk2gt_1,115,53,549).  visible(wk2gt_1,549). happensAt(inactive(wk2gt_1),549). orientation(wk2gt_4,149,549). coords(wk2gt_4,49,101,549).  visible(wk2gt_4,549). happensAt(walking(wk2gt_4),549).
orientation(wk2gt_1,122,550). coords(wk2gt_1,115,53,550).  visible(wk2gt_1,550). happensAt(inactive(wk2gt_1),550). orientation(wk2gt_4,149,550). coords(wk2gt_4,49,101,550).  visible(wk2gt_4,550). happensAt(walking(wk2gt_4),550).
orientation(wk2gt_1,122,551). coords(wk2gt_1,115,53,551).  visible(wk2gt_1,551). happensAt(inactive(wk2gt_1),551). orientation(wk2gt_4,149,551). coords(wk2gt_4,46,101,551).  visible(wk2gt_4,551). happensAt(walking(wk2gt_4),551).
orientation(wk2gt_1,122,552). coords(wk2gt_1,115,53,552).  visible(wk2gt_1,552). happensAt(inactive(wk2gt_1),552). orientation(wk2gt_4,149,552). coords(wk2gt_4,46,101,552).  visible(wk2gt_4,552). happensAt(walking(wk2gt_4),552).
orientation(wk2gt_1,122,553). coords(wk2gt_1,115,53,553).  visible(wk2gt_1,553). happensAt(inactive(wk2gt_1),553). orientation(wk2gt_4,149,553). coords(wk2gt_4,45,101,553).  visible(wk2gt_4,553). happensAt(walking(wk2gt_4),553).
orientation(wk2gt_1,122,554). coords(wk2gt_1,115,53,554).  visible(wk2gt_1,554). happensAt(inactive(wk2gt_1),554). orientation(wk2gt_4,149,554). coords(wk2gt_4,45,101,554).  visible(wk2gt_4,554). happensAt(walking(wk2gt_4),554).
orientation(wk2gt_1,122,555). coords(wk2gt_1,115,53,555).  visible(wk2gt_1,555). happensAt(inactive(wk2gt_1),555). orientation(wk2gt_4,149,555). coords(wk2gt_4,45,101,555).  visible(wk2gt_4,555). happensAt(walking(wk2gt_4),555).
orientation(wk2gt_1,122,556). coords(wk2gt_1,115,53,556).  visible(wk2gt_1,556). happensAt(inactive(wk2gt_1),556). orientation(wk2gt_4,147,556). coords(wk2gt_4,45,103,556).  visible(wk2gt_4,556). happensAt(walking(wk2gt_4),556).
orientation(wk2gt_1,122,557). coords(wk2gt_1,115,53,557).  visible(wk2gt_1,557). happensAt(inactive(wk2gt_1),557). orientation(wk2gt_4,147,557). coords(wk2gt_4,44,103,557).  visible(wk2gt_4,557). happensAt(walking(wk2gt_4),557).
orientation(wk2gt_1,122,558). coords(wk2gt_1,115,53,558).  visible(wk2gt_1,558). happensAt(inactive(wk2gt_1),558). orientation(wk2gt_4,147,558). coords(wk2gt_4,44,103,558).  visible(wk2gt_4,558). happensAt(walking(wk2gt_4),558).
orientation(wk2gt_1,122,559). coords(wk2gt_1,115,53,559).  visible(wk2gt_1,559). happensAt(inactive(wk2gt_1),559). orientation(wk2gt_4,147,559). coords(wk2gt_4,44,104,559).  visible(wk2gt_4,559). happensAt(walking(wk2gt_4),559).
#show holdsAt/2.