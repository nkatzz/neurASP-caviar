
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

	orientation(rwgt_0,137,534). coords(rwgt_0,316,283,534).  visible(rwgt_0,534). happensAt(inactive(rwgt_0),534). orientation(rwgt_2,132,534). coords(rwgt_2,155,35,534).  visible(rwgt_2,534). happensAt(walking(rwgt_2),534).
orientation(rwgt_0,137,535). coords(rwgt_0,316,283,535).  visible(rwgt_0,535). happensAt(inactive(rwgt_0),535). orientation(rwgt_2,132,535). coords(rwgt_2,156,35,535).  visible(rwgt_2,535). happensAt(walking(rwgt_2),535).
orientation(rwgt_0,137,536). coords(rwgt_0,316,283,536).  visible(rwgt_0,536). happensAt(inactive(rwgt_0),536). orientation(rwgt_2,132,536). coords(rwgt_2,156,35,536).  visible(rwgt_2,536). happensAt(walking(rwgt_2),536).
orientation(rwgt_0,137,537). coords(rwgt_0,316,283,537).  visible(rwgt_0,537). happensAt(inactive(rwgt_0),537). orientation(rwgt_2,132,537). coords(rwgt_2,156,35,537).  visible(rwgt_2,537). happensAt(walking(rwgt_2),537).
orientation(rwgt_0,137,538). coords(rwgt_0,316,283,538).  visible(rwgt_0,538). happensAt(inactive(rwgt_0),538). orientation(rwgt_2,132,538). coords(rwgt_2,155,35,538).  visible(rwgt_2,538). happensAt(walking(rwgt_2),538).
orientation(rwgt_0,137,539). coords(rwgt_0,316,283,539).  visible(rwgt_0,539). happensAt(inactive(rwgt_0),539). orientation(rwgt_2,137,539). coords(rwgt_2,155,35,539).  visible(rwgt_2,539). happensAt(walking(rwgt_2),539).
orientation(rwgt_0,137,540). coords(rwgt_0,316,283,540).  visible(rwgt_0,540). happensAt(inactive(rwgt_0),540). orientation(rwgt_2,137,540). coords(rwgt_2,155,35,540).  visible(rwgt_2,540). happensAt(walking(rwgt_2),540).
orientation(rwgt_0,137,541). coords(rwgt_0,316,283,541).  visible(rwgt_0,541). happensAt(inactive(rwgt_0),541). orientation(rwgt_2,137,541). coords(rwgt_2,155,35,541).  visible(rwgt_2,541). happensAt(walking(rwgt_2),541).
orientation(rwgt_0,137,542). coords(rwgt_0,316,283,542).  visible(rwgt_0,542). happensAt(inactive(rwgt_0),542). orientation(rwgt_2,137,542). coords(rwgt_2,155,35,542).  visible(rwgt_2,542). happensAt(walking(rwgt_2),542).
orientation(rwgt_0,137,543). coords(rwgt_0,316,283,543).  visible(rwgt_0,543). happensAt(inactive(rwgt_0),543). orientation(rwgt_2,137,543). coords(rwgt_2,155,35,543).  visible(rwgt_2,543). happensAt(walking(rwgt_2),543).
orientation(rwgt_0,137,544). coords(rwgt_0,316,283,544).  visible(rwgt_0,544). happensAt(inactive(rwgt_0),544). orientation(rwgt_2,137,544). coords(rwgt_2,155,36,544).  visible(rwgt_2,544). happensAt(walking(rwgt_2),544).
orientation(rwgt_0,137,545). coords(rwgt_0,316,283,545).  visible(rwgt_0,545). happensAt(inactive(rwgt_0),545). orientation(rwgt_2,137,545). coords(rwgt_2,154,37,545).  visible(rwgt_2,545). happensAt(walking(rwgt_2),545).
orientation(rwgt_0,137,546). coords(rwgt_0,316,283,546).  visible(rwgt_0,546). happensAt(inactive(rwgt_0),546). orientation(rwgt_2,137,546). coords(rwgt_2,154,37,546).  visible(rwgt_2,546). happensAt(walking(rwgt_2),546).
orientation(rwgt_0,137,547). coords(rwgt_0,316,283,547).  visible(rwgt_0,547). happensAt(inactive(rwgt_0),547). orientation(rwgt_2,137,547). coords(rwgt_2,154,37,547).  visible(rwgt_2,547). happensAt(walking(rwgt_2),547).
orientation(rwgt_0,137,548). coords(rwgt_0,316,283,548).  visible(rwgt_0,548). happensAt(inactive(rwgt_0),548). orientation(rwgt_2,137,548). coords(rwgt_2,154,37,548).  visible(rwgt_2,548). happensAt(walking(rwgt_2),548).
orientation(rwgt_0,137,549). coords(rwgt_0,316,283,549).  visible(rwgt_0,549). happensAt(inactive(rwgt_0),549). orientation(rwgt_2,137,549). coords(rwgt_2,153,37,549).  visible(rwgt_2,549). happensAt(walking(rwgt_2),549).
orientation(rwgt_0,137,550). coords(rwgt_0,316,283,550).  visible(rwgt_0,550). happensAt(inactive(rwgt_0),550). orientation(rwgt_2,137,550). coords(rwgt_2,153,37,550).  visible(rwgt_2,550). happensAt(walking(rwgt_2),550).
orientation(rwgt_0,137,551). coords(rwgt_0,316,283,551).  visible(rwgt_0,551). happensAt(inactive(rwgt_0),551). orientation(rwgt_2,137,551). coords(rwgt_2,152,37,551).  visible(rwgt_2,551). happensAt(walking(rwgt_2),551).
orientation(rwgt_0,137,552). coords(rwgt_0,316,283,552).  visible(rwgt_0,552). happensAt(inactive(rwgt_0),552). orientation(rwgt_2,144,552). coords(rwgt_2,151,38,552).  visible(rwgt_2,552). happensAt(walking(rwgt_2),552).
orientation(rwgt_0,137,553). coords(rwgt_0,316,283,553).  visible(rwgt_0,553). happensAt(inactive(rwgt_0),553). orientation(rwgt_2,144,553). coords(rwgt_2,150,38,553).  visible(rwgt_2,553). happensAt(walking(rwgt_2),553).
orientation(rwgt_0,137,554). coords(rwgt_0,316,283,554).  visible(rwgt_0,554). happensAt(inactive(rwgt_0),554). orientation(rwgt_2,144,554). coords(rwgt_2,150,38,554).  visible(rwgt_2,554). happensAt(walking(rwgt_2),554).
orientation(rwgt_0,137,555). coords(rwgt_0,316,283,555).  visible(rwgt_0,555). happensAt(inactive(rwgt_0),555). orientation(rwgt_2,144,555). coords(rwgt_2,149,38,555).  visible(rwgt_2,555). happensAt(walking(rwgt_2),555).
orientation(rwgt_0,137,556). coords(rwgt_0,316,283,556).  visible(rwgt_0,556). happensAt(inactive(rwgt_0),556). orientation(rwgt_2,144,556). coords(rwgt_2,149,39,556).  visible(rwgt_2,556). happensAt(walking(rwgt_2),556).
orientation(rwgt_0,137,557). coords(rwgt_0,316,283,557).  visible(rwgt_0,557). happensAt(inactive(rwgt_0),557). orientation(rwgt_2,144,557). coords(rwgt_2,149,39,557).  visible(rwgt_2,557). happensAt(walking(rwgt_2),557).
orientation(rwgt_0,137,558). coords(rwgt_0,316,283,558).  visible(rwgt_0,558). happensAt(inactive(rwgt_0),558). orientation(rwgt_2,144,558). coords(rwgt_2,148,39,558).  visible(rwgt_2,558). happensAt(walking(rwgt_2),558).
#show holdsAt/2.