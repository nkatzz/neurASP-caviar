
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

	orientation(ms3ggt_0,168,536). coords(ms3ggt_0,73,191,536).  visible(ms3ggt_0,536). happensAt(inactive(ms3ggt_0),536). orientation(ms3ggt_2,151,536). coords(ms3ggt_2,85,103,536).  visible(ms3ggt_2,536). happensAt(walking(ms3ggt_2),536).
orientation(ms3ggt_0,168,537). coords(ms3ggt_0,73,191,537).  visible(ms3ggt_0,537). happensAt(inactive(ms3ggt_0),537). orientation(ms3ggt_2,151,537). coords(ms3ggt_2,85,102,537).  visible(ms3ggt_2,537). happensAt(walking(ms3ggt_2),537).
orientation(ms3ggt_0,168,538). coords(ms3ggt_0,73,191,538).  visible(ms3ggt_0,538). happensAt(inactive(ms3ggt_0),538). orientation(ms3ggt_2,151,538). coords(ms3ggt_2,85,102,538).  visible(ms3ggt_2,538). happensAt(walking(ms3ggt_2),538).
orientation(ms3ggt_0,168,539). coords(ms3ggt_0,73,191,539).  visible(ms3ggt_0,539). happensAt(inactive(ms3ggt_0),539). orientation(ms3ggt_2,151,539). coords(ms3ggt_2,85,102,539).  visible(ms3ggt_2,539). happensAt(walking(ms3ggt_2),539).
orientation(ms3ggt_0,168,540). coords(ms3ggt_0,73,191,540).  visible(ms3ggt_0,540). happensAt(inactive(ms3ggt_0),540). orientation(ms3ggt_2,151,540). coords(ms3ggt_2,85,102,540).  visible(ms3ggt_2,540). happensAt(walking(ms3ggt_2),540).
orientation(ms3ggt_0,168,541). coords(ms3ggt_0,73,191,541).  visible(ms3ggt_0,541). happensAt(inactive(ms3ggt_0),541). orientation(ms3ggt_2,139,541). coords(ms3ggt_2,85,102,541).  visible(ms3ggt_2,541). happensAt(walking(ms3ggt_2),541).
orientation(ms3ggt_0,168,542). coords(ms3ggt_0,73,191,542).  visible(ms3ggt_0,542). happensAt(inactive(ms3ggt_0),542). orientation(ms3ggt_2,139,542). coords(ms3ggt_2,84,101,542).  visible(ms3ggt_2,542). happensAt(walking(ms3ggt_2),542).
orientation(ms3ggt_0,168,543). coords(ms3ggt_0,73,191,543).  visible(ms3ggt_0,543). happensAt(inactive(ms3ggt_0),543). orientation(ms3ggt_2,139,543). coords(ms3ggt_2,84,100,543).  visible(ms3ggt_2,543). happensAt(walking(ms3ggt_2),543).
orientation(ms3ggt_0,168,544). coords(ms3ggt_0,73,191,544).  visible(ms3ggt_0,544). happensAt(inactive(ms3ggt_0),544). orientation(ms3ggt_2,139,544). coords(ms3ggt_2,84,100,544).  visible(ms3ggt_2,544). happensAt(walking(ms3ggt_2),544).
orientation(ms3ggt_0,168,545). coords(ms3ggt_0,73,191,545).  visible(ms3ggt_0,545). happensAt(inactive(ms3ggt_0),545). orientation(ms3ggt_2,139,545). coords(ms3ggt_2,84,99,545).  visible(ms3ggt_2,545). happensAt(walking(ms3ggt_2),545).
orientation(ms3ggt_0,168,546). coords(ms3ggt_0,73,191,546).  visible(ms3ggt_0,546). happensAt(inactive(ms3ggt_0),546). orientation(ms3ggt_2,139,546). coords(ms3ggt_2,84,99,546).  visible(ms3ggt_2,546). happensAt(walking(ms3ggt_2),546).
orientation(ms3ggt_0,168,547). coords(ms3ggt_0,73,191,547).  visible(ms3ggt_0,547). happensAt(inactive(ms3ggt_0),547). orientation(ms3ggt_2,142,547). coords(ms3ggt_2,84,98,547).  visible(ms3ggt_2,547). happensAt(walking(ms3ggt_2),547).
orientation(ms3ggt_0,168,548). coords(ms3ggt_0,73,191,548).  visible(ms3ggt_0,548). happensAt(inactive(ms3ggt_0),548). orientation(ms3ggt_2,142,548). coords(ms3ggt_2,84,96,548).  visible(ms3ggt_2,548). happensAt(walking(ms3ggt_2),548).
orientation(ms3ggt_0,168,549). coords(ms3ggt_0,73,191,549).  visible(ms3ggt_0,549). happensAt(inactive(ms3ggt_0),549). orientation(ms3ggt_2,142,549). coords(ms3ggt_2,84,96,549).  visible(ms3ggt_2,549). happensAt(walking(ms3ggt_2),549).
orientation(ms3ggt_0,168,550). coords(ms3ggt_0,73,191,550).  visible(ms3ggt_0,550). happensAt(inactive(ms3ggt_0),550). orientation(ms3ggt_2,142,550). coords(ms3ggt_2,84,96,550).  visible(ms3ggt_2,550). happensAt(walking(ms3ggt_2),550).
orientation(ms3ggt_0,168,551). coords(ms3ggt_0,73,191,551).  visible(ms3ggt_0,551). happensAt(inactive(ms3ggt_0),551). orientation(ms3ggt_2,142,551). coords(ms3ggt_2,84,96,551).  visible(ms3ggt_2,551). happensAt(walking(ms3ggt_2),551).
orientation(ms3ggt_0,168,552). coords(ms3ggt_0,73,191,552).  visible(ms3ggt_0,552). happensAt(inactive(ms3ggt_0),552). orientation(ms3ggt_2,142,552). coords(ms3ggt_2,84,95,552).  visible(ms3ggt_2,552). happensAt(walking(ms3ggt_2),552).
orientation(ms3ggt_0,168,553). coords(ms3ggt_0,73,191,553).  visible(ms3ggt_0,553). happensAt(inactive(ms3ggt_0),553). orientation(ms3ggt_2,142,553). coords(ms3ggt_2,84,95,553).  visible(ms3ggt_2,553). happensAt(walking(ms3ggt_2),553).
orientation(ms3ggt_0,168,554). coords(ms3ggt_0,73,191,554).  visible(ms3ggt_0,554). happensAt(inactive(ms3ggt_0),554). orientation(ms3ggt_2,142,554). coords(ms3ggt_2,84,95,554).  visible(ms3ggt_2,554). happensAt(walking(ms3ggt_2),554).
orientation(ms3ggt_0,168,555). coords(ms3ggt_0,73,191,555).  visible(ms3ggt_0,555). happensAt(inactive(ms3ggt_0),555). orientation(ms3ggt_2,142,555). coords(ms3ggt_2,84,95,555).  visible(ms3ggt_2,555). happensAt(walking(ms3ggt_2),555).
orientation(ms3ggt_0,168,556). coords(ms3ggt_0,73,191,556).  visible(ms3ggt_0,556). happensAt(inactive(ms3ggt_0),556). orientation(ms3ggt_2,142,556). coords(ms3ggt_2,84,94,556).  visible(ms3ggt_2,556). happensAt(walking(ms3ggt_2),556).
orientation(ms3ggt_0,168,557). coords(ms3ggt_0,73,191,557).  visible(ms3ggt_0,557). happensAt(inactive(ms3ggt_0),557). orientation(ms3ggt_2,142,557). coords(ms3ggt_2,84,94,557).  visible(ms3ggt_2,557). happensAt(walking(ms3ggt_2),557).
orientation(ms3ggt_0,168,558). coords(ms3ggt_0,73,191,558).  visible(ms3ggt_0,558). happensAt(inactive(ms3ggt_0),558). orientation(ms3ggt_2,142,558). coords(ms3ggt_2,84,93,558).  visible(ms3ggt_2,558). happensAt(walking(ms3ggt_2),558).
orientation(ms3ggt_0,168,559). coords(ms3ggt_0,73,191,559).  visible(ms3ggt_0,559). happensAt(inactive(ms3ggt_0),559). orientation(ms3ggt_2,142,559). coords(ms3ggt_2,84,93,559).  visible(ms3ggt_2,559). happensAt(walking(ms3ggt_2),559).
orientation(ms3ggt_0,168,560). coords(ms3ggt_0,73,191,560).  visible(ms3ggt_0,560). happensAt(inactive(ms3ggt_0),560). orientation(ms3ggt_2,150,560). coords(ms3ggt_2,84,92,560).  visible(ms3ggt_2,560). happensAt(walking(ms3ggt_2),560).
#show holdsAt/2.