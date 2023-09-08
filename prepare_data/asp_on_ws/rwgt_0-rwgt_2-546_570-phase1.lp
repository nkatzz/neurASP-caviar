
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
orientation(rwgt_0,137,559). coords(rwgt_0,316,283,559).  visible(rwgt_0,559). happensAt(inactive(rwgt_0),559). orientation(rwgt_2,144,559). coords(rwgt_2,148,39,559).  visible(rwgt_2,559). happensAt(walking(rwgt_2),559).
orientation(rwgt_0,137,560). coords(rwgt_0,316,283,560).  visible(rwgt_0,560). happensAt(inactive(rwgt_0),560). orientation(rwgt_2,144,560). coords(rwgt_2,148,40,560).  visible(rwgt_2,560). happensAt(walking(rwgt_2),560).
orientation(rwgt_0,137,561). coords(rwgt_0,316,283,561).  visible(rwgt_0,561). happensAt(inactive(rwgt_0),561). orientation(rwgt_2,144,561). coords(rwgt_2,148,40,561).  visible(rwgt_2,561). happensAt(walking(rwgt_2),561).
orientation(rwgt_0,137,562). coords(rwgt_0,316,283,562).  visible(rwgt_0,562). happensAt(inactive(rwgt_0),562). orientation(rwgt_2,144,562). coords(rwgt_2,148,40,562).  visible(rwgt_2,562). happensAt(walking(rwgt_2),562).
orientation(rwgt_0,137,563). coords(rwgt_0,316,283,563).  visible(rwgt_0,563). happensAt(inactive(rwgt_0),563). orientation(rwgt_2,144,563). coords(rwgt_2,147,40,563).  visible(rwgt_2,563). happensAt(walking(rwgt_2),563).
orientation(rwgt_0,137,564). coords(rwgt_0,316,283,564).  visible(rwgt_0,564). happensAt(inactive(rwgt_0),564). orientation(rwgt_2,144,564). coords(rwgt_2,147,40,564).  visible(rwgt_2,564). happensAt(walking(rwgt_2),564).
orientation(rwgt_0,137,565). coords(rwgt_0,316,283,565).  visible(rwgt_0,565). happensAt(inactive(rwgt_0),565). orientation(rwgt_2,144,565). coords(rwgt_2,147,40,565).  visible(rwgt_2,565). happensAt(walking(rwgt_2),565).
orientation(rwgt_0,137,566). coords(rwgt_0,316,283,566).  visible(rwgt_0,566). happensAt(inactive(rwgt_0),566). orientation(rwgt_2,144,566). coords(rwgt_2,147,40,566).  visible(rwgt_2,566). happensAt(walking(rwgt_2),566).
orientation(rwgt_0,137,567). coords(rwgt_0,316,283,567).  visible(rwgt_0,567). happensAt(inactive(rwgt_0),567). orientation(rwgt_2,144,567). coords(rwgt_2,147,40,567).  visible(rwgt_2,567). happensAt(walking(rwgt_2),567).
orientation(rwgt_0,137,568). coords(rwgt_0,316,283,568).  visible(rwgt_0,568). happensAt(inactive(rwgt_0),568). orientation(rwgt_2,144,568). coords(rwgt_2,147,41,568).  visible(rwgt_2,568). happensAt(walking(rwgt_2),568).
orientation(rwgt_0,137,569). coords(rwgt_0,316,283,569).  visible(rwgt_0,569). happensAt(inactive(rwgt_0),569). orientation(rwgt_2,144,569). coords(rwgt_2,146,42,569).  visible(rwgt_2,569). happensAt(walking(rwgt_2),569).
orientation(rwgt_0,137,570). coords(rwgt_0,316,283,570).  visible(rwgt_0,570). happensAt(inactive(rwgt_0),570). orientation(rwgt_2,144,570). coords(rwgt_2,146,42,570).  visible(rwgt_2,570). happensAt(walking(rwgt_2),570).
#show holdsAt/2.