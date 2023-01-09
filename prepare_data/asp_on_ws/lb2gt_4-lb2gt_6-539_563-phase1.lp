
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

	orientation(lb2gt_4,154,539). coords(lb2gt_4,103,62,539).  visible(lb2gt_4,539). happensAt(walking(lb2gt_4),539). orientation(lb2gt_6,151,539). coords(lb2gt_6,72,122,539).  visible(lb2gt_6,539). happensAt(inactive(lb2gt_6),539).
orientation(lb2gt_4,154,540). coords(lb2gt_4,104,60,540).  visible(lb2gt_4,540). happensAt(walking(lb2gt_4),540). orientation(lb2gt_6,151,540). coords(lb2gt_6,72,122,540).  visible(lb2gt_6,540). happensAt(inactive(lb2gt_6),540).
orientation(lb2gt_4,154,541). coords(lb2gt_4,104,60,541).  visible(lb2gt_4,541). happensAt(walking(lb2gt_4),541). orientation(lb2gt_6,151,541). coords(lb2gt_6,72,122,541).  visible(lb2gt_6,541). happensAt(inactive(lb2gt_6),541).
orientation(lb2gt_4,154,542). coords(lb2gt_4,104,58,542).  visible(lb2gt_4,542). happensAt(walking(lb2gt_4),542). orientation(lb2gt_6,151,542). coords(lb2gt_6,72,122,542).  visible(lb2gt_6,542). happensAt(inactive(lb2gt_6),542).
orientation(lb2gt_4,154,543). coords(lb2gt_4,104,58,543).  visible(lb2gt_4,543). happensAt(walking(lb2gt_4),543). orientation(lb2gt_6,151,543). coords(lb2gt_6,72,122,543).  visible(lb2gt_6,543). happensAt(inactive(lb2gt_6),543).
orientation(lb2gt_4,154,544). coords(lb2gt_4,105,57,544).  visible(lb2gt_4,544). happensAt(walking(lb2gt_4),544). orientation(lb2gt_6,151,544). coords(lb2gt_6,72,122,544).  visible(lb2gt_6,544). happensAt(inactive(lb2gt_6),544).
orientation(lb2gt_4,154,545). coords(lb2gt_4,107,56,545).  visible(lb2gt_4,545). happensAt(walking(lb2gt_4),545). orientation(lb2gt_6,151,545). coords(lb2gt_6,72,122,545).  visible(lb2gt_6,545). happensAt(inactive(lb2gt_6),545).
orientation(lb2gt_4,154,546). coords(lb2gt_4,108,56,546).  visible(lb2gt_4,546). happensAt(walking(lb2gt_4),546). orientation(lb2gt_6,151,546). coords(lb2gt_6,72,122,546).  visible(lb2gt_6,546). happensAt(inactive(lb2gt_6),546).
orientation(lb2gt_4,154,547). coords(lb2gt_4,110,56,547).  visible(lb2gt_4,547). happensAt(walking(lb2gt_4),547). orientation(lb2gt_6,151,547). coords(lb2gt_6,72,122,547).  visible(lb2gt_6,547). happensAt(inactive(lb2gt_6),547).
orientation(lb2gt_4,154,548). coords(lb2gt_4,111,56,548).  visible(lb2gt_4,548). happensAt(walking(lb2gt_4),548). orientation(lb2gt_6,151,548). coords(lb2gt_6,72,122,548).  visible(lb2gt_6,548). happensAt(inactive(lb2gt_6),548).
orientation(lb2gt_4,154,549). coords(lb2gt_4,113,56,549).  visible(lb2gt_4,549). happensAt(walking(lb2gt_4),549). orientation(lb2gt_6,151,549). coords(lb2gt_6,72,122,549).  visible(lb2gt_6,549). happensAt(inactive(lb2gt_6),549).
orientation(lb2gt_4,154,550). coords(lb2gt_4,113,55,550).  visible(lb2gt_4,550). happensAt(walking(lb2gt_4),550). orientation(lb2gt_6,151,550). coords(lb2gt_6,72,122,550).  visible(lb2gt_6,550). happensAt(inactive(lb2gt_6),550).
orientation(lb2gt_4,154,551). coords(lb2gt_4,113,54,551).  visible(lb2gt_4,551). happensAt(walking(lb2gt_4),551). orientation(lb2gt_6,151,551). coords(lb2gt_6,72,122,551).  visible(lb2gt_6,551). happensAt(inactive(lb2gt_6),551).
orientation(lb2gt_4,154,552). coords(lb2gt_4,114,53,552).  visible(lb2gt_4,552). happensAt(walking(lb2gt_4),552). orientation(lb2gt_6,151,552). coords(lb2gt_6,72,122,552).  visible(lb2gt_6,552). happensAt(inactive(lb2gt_6),552).
orientation(lb2gt_4,154,553). coords(lb2gt_4,114,52,553).  visible(lb2gt_4,553). happensAt(walking(lb2gt_4),553). orientation(lb2gt_6,151,553). coords(lb2gt_6,72,122,553).  visible(lb2gt_6,553). happensAt(inactive(lb2gt_6),553).
orientation(lb2gt_4,154,554). coords(lb2gt_4,115,52,554).  visible(lb2gt_4,554). happensAt(walking(lb2gt_4),554). orientation(lb2gt_6,151,554). coords(lb2gt_6,72,122,554).  visible(lb2gt_6,554). happensAt(inactive(lb2gt_6),554).
orientation(lb2gt_4,154,555). coords(lb2gt_4,115,51,555).  visible(lb2gt_4,555). happensAt(walking(lb2gt_4),555). orientation(lb2gt_6,151,555). coords(lb2gt_6,72,122,555).  visible(lb2gt_6,555). happensAt(inactive(lb2gt_6),555).
orientation(lb2gt_4,154,556). coords(lb2gt_4,116,51,556).  visible(lb2gt_4,556). happensAt(walking(lb2gt_4),556). orientation(lb2gt_6,151,556). coords(lb2gt_6,72,122,556).  visible(lb2gt_6,556). happensAt(inactive(lb2gt_6),556).
orientation(lb2gt_4,154,557). coords(lb2gt_4,117,49,557).  visible(lb2gt_4,557). happensAt(walking(lb2gt_4),557). orientation(lb2gt_6,151,557). coords(lb2gt_6,72,122,557).  visible(lb2gt_6,557). happensAt(inactive(lb2gt_6),557).
orientation(lb2gt_4,154,558). coords(lb2gt_4,118,49,558).  visible(lb2gt_4,558). happensAt(walking(lb2gt_4),558). orientation(lb2gt_6,151,558). coords(lb2gt_6,72,122,558).  visible(lb2gt_6,558). happensAt(inactive(lb2gt_6),558).
orientation(lb2gt_4,154,559). coords(lb2gt_4,119,49,559).  visible(lb2gt_4,559). happensAt(walking(lb2gt_4),559). orientation(lb2gt_6,151,559). coords(lb2gt_6,72,122,559).  visible(lb2gt_6,559). happensAt(inactive(lb2gt_6),559).
orientation(lb2gt_4,154,560). coords(lb2gt_4,120,48,560).  visible(lb2gt_4,560). happensAt(walking(lb2gt_4),560). orientation(lb2gt_6,151,560). coords(lb2gt_6,72,122,560).  visible(lb2gt_6,560). happensAt(inactive(lb2gt_6),560).
orientation(lb2gt_4,154,561). coords(lb2gt_4,122,47,561).  visible(lb2gt_4,561). happensAt(walking(lb2gt_4),561). orientation(lb2gt_6,151,561). coords(lb2gt_6,72,122,561).  visible(lb2gt_6,561). happensAt(inactive(lb2gt_6),561).
orientation(lb2gt_4,154,562). coords(lb2gt_4,124,47,562).  visible(lb2gt_4,562). happensAt(walking(lb2gt_4),562). orientation(lb2gt_6,151,562). coords(lb2gt_6,72,122,562).  visible(lb2gt_6,562). happensAt(inactive(lb2gt_6),562).
orientation(lb2gt_4,154,563). coords(lb2gt_4,124,47,563).  visible(lb2gt_4,563). happensAt(walking(lb2gt_4),563). orientation(lb2gt_6,151,563). coords(lb2gt_6,72,122,563).  visible(lb2gt_6,563). happensAt(inactive(lb2gt_6),563).
#show holdsAt/2.