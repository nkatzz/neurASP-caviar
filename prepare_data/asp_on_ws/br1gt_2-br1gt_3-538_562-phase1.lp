
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

	orientation(br1gt_2,12,538). coords(br1gt_2,86,209,538).  visible(br1gt_2,538). happensAt(inactive(br1gt_2),538). orientation(br1gt_3,152,538). coords(br1gt_3,160,119,538).  visible(br1gt_3,538). happensAt(walking(br1gt_3),538).
orientation(br1gt_2,12,539). coords(br1gt_2,86,209,539).  visible(br1gt_2,539). happensAt(inactive(br1gt_2),539). orientation(br1gt_3,152,539). coords(br1gt_3,163,119,539).  visible(br1gt_3,539). happensAt(walking(br1gt_3),539).
orientation(br1gt_2,12,540). coords(br1gt_2,86,209,540).  visible(br1gt_2,540). happensAt(inactive(br1gt_2),540). orientation(br1gt_3,152,540). coords(br1gt_3,165,119,540).  visible(br1gt_3,540). happensAt(walking(br1gt_3),540).
orientation(br1gt_2,12,541). coords(br1gt_2,86,209,541).  visible(br1gt_2,541). happensAt(inactive(br1gt_2),541). orientation(br1gt_3,152,541). coords(br1gt_3,168,118,541).  visible(br1gt_3,541). happensAt(walking(br1gt_3),541).
orientation(br1gt_2,12,542). coords(br1gt_2,86,209,542).  visible(br1gt_2,542). happensAt(inactive(br1gt_2),542). orientation(br1gt_3,152,542). coords(br1gt_3,170,118,542).  visible(br1gt_3,542). happensAt(walking(br1gt_3),542).
orientation(br1gt_2,12,543). coords(br1gt_2,86,209,543).  visible(br1gt_2,543). happensAt(inactive(br1gt_2),543). orientation(br1gt_3,154,543). coords(br1gt_3,172,117,543).  visible(br1gt_3,543). happensAt(walking(br1gt_3),543).
orientation(br1gt_2,12,544). coords(br1gt_2,86,209,544).  visible(br1gt_2,544). happensAt(inactive(br1gt_2),544). orientation(br1gt_3,154,544). coords(br1gt_3,173,117,544).  visible(br1gt_3,544). happensAt(walking(br1gt_3),544).
orientation(br1gt_2,12,545). coords(br1gt_2,86,209,545).  visible(br1gt_2,545). happensAt(inactive(br1gt_2),545). orientation(br1gt_3,150,545). coords(br1gt_3,174,116,545).  visible(br1gt_3,545). happensAt(walking(br1gt_3),545).
orientation(br1gt_2,12,546). coords(br1gt_2,86,209,546).  visible(br1gt_2,546). happensAt(inactive(br1gt_2),546). orientation(br1gt_3,150,546). coords(br1gt_3,175,115,546).  visible(br1gt_3,546). happensAt(walking(br1gt_3),546).
orientation(br1gt_2,12,547). coords(br1gt_2,86,209,547).  visible(br1gt_2,547). happensAt(inactive(br1gt_2),547). orientation(br1gt_3,150,547). coords(br1gt_3,176,114,547).  visible(br1gt_3,547). happensAt(walking(br1gt_3),547).
orientation(br1gt_2,12,548). coords(br1gt_2,86,209,548).  visible(br1gt_2,548). happensAt(inactive(br1gt_2),548). orientation(br1gt_3,150,548). coords(br1gt_3,177,114,548).  visible(br1gt_3,548). happensAt(walking(br1gt_3),548).
orientation(br1gt_2,12,549). coords(br1gt_2,86,209,549).  visible(br1gt_2,549). happensAt(inactive(br1gt_2),549). orientation(br1gt_3,145,549). coords(br1gt_3,177,113,549).  visible(br1gt_3,549). happensAt(walking(br1gt_3),549).
orientation(br1gt_2,12,550). coords(br1gt_2,86,209,550).  visible(br1gt_2,550). happensAt(inactive(br1gt_2),550). orientation(br1gt_3,142,550). coords(br1gt_3,178,112,550).  visible(br1gt_3,550). happensAt(walking(br1gt_3),550).
orientation(br1gt_2,12,551). coords(br1gt_2,86,209,551).  visible(br1gt_2,551). happensAt(inactive(br1gt_2),551). orientation(br1gt_3,142,551). coords(br1gt_3,182,112,551).  visible(br1gt_3,551). happensAt(walking(br1gt_3),551).
orientation(br1gt_2,12,552). coords(br1gt_2,86,209,552).  visible(br1gt_2,552). happensAt(inactive(br1gt_2),552). orientation(br1gt_3,142,552). coords(br1gt_3,186,111,552).  visible(br1gt_3,552). happensAt(walking(br1gt_3),552).
orientation(br1gt_2,12,553). coords(br1gt_2,86,209,553).  visible(br1gt_2,553). happensAt(inactive(br1gt_2),553). orientation(br1gt_3,142,553). coords(br1gt_3,189,111,553).  visible(br1gt_3,553). happensAt(walking(br1gt_3),553).
orientation(br1gt_2,12,554). coords(br1gt_2,86,209,554).  visible(br1gt_2,554). happensAt(inactive(br1gt_2),554). orientation(br1gt_3,148,554). coords(br1gt_3,191,109,554).  visible(br1gt_3,554). happensAt(walking(br1gt_3),554).
orientation(br1gt_2,12,555). coords(br1gt_2,86,209,555).  visible(br1gt_2,555). happensAt(inactive(br1gt_2),555). orientation(br1gt_3,148,555). coords(br1gt_3,193,110,555).  visible(br1gt_3,555). happensAt(walking(br1gt_3),555).
orientation(br1gt_2,12,556). coords(br1gt_2,86,209,556).  visible(br1gt_2,556). happensAt(inactive(br1gt_2),556). orientation(br1gt_3,148,556). coords(br1gt_3,196,110,556).  visible(br1gt_3,556). happensAt(walking(br1gt_3),556).
orientation(br1gt_2,12,557). coords(br1gt_2,86,209,557).  visible(br1gt_2,557). happensAt(inactive(br1gt_2),557). orientation(br1gt_3,148,557). coords(br1gt_3,197,110,557).  visible(br1gt_3,557). happensAt(walking(br1gt_3),557).
orientation(br1gt_2,12,558). coords(br1gt_2,86,209,558).  visible(br1gt_2,558). happensAt(inactive(br1gt_2),558). orientation(br1gt_3,148,558). coords(br1gt_3,199,110,558).  visible(br1gt_3,558). happensAt(walking(br1gt_3),558).
orientation(br1gt_2,12,559). coords(br1gt_2,86,209,559).  visible(br1gt_2,559). happensAt(inactive(br1gt_2),559). orientation(br1gt_3,148,559). coords(br1gt_3,199,109,559).  visible(br1gt_3,559). happensAt(walking(br1gt_3),559).
orientation(br1gt_2,12,560). coords(br1gt_2,86,209,560).  visible(br1gt_2,560). happensAt(inactive(br1gt_2),560). orientation(br1gt_3,148,560). coords(br1gt_3,200,109,560).  visible(br1gt_3,560). happensAt(walking(br1gt_3),560).
orientation(br1gt_2,12,561). coords(br1gt_2,86,209,561).  visible(br1gt_2,561). happensAt(inactive(br1gt_2),561). orientation(br1gt_3,148,561). coords(br1gt_3,201,109,561).  visible(br1gt_3,561). happensAt(walking(br1gt_3),561).
orientation(br1gt_2,12,562). coords(br1gt_2,86,209,562).  visible(br1gt_2,562). happensAt(inactive(br1gt_2),562). orientation(br1gt_3,145,562). coords(br1gt_3,202,108,562).  visible(br1gt_3,562). happensAt(walking(br1gt_3),562).
#show holdsAt/2.