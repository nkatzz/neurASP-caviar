
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

	orientation(fomdgt3_2,110,542). coords(fomdgt3_2,236,94,542).  visible(fomdgt3_2,542). happensAt(walking(fomdgt3_2),542). orientation(fomdgt3_7,159,542). coords(fomdgt3_7,172,213,542).  visible(fomdgt3_7,542). happensAt(walking(fomdgt3_7),542).
orientation(fomdgt3_2,110,543). coords(fomdgt3_2,236,94,543).  visible(fomdgt3_2,543). happensAt(walking(fomdgt3_2),543). orientation(fomdgt3_7,170,543). coords(fomdgt3_7,174,214,543).  visible(fomdgt3_7,543). happensAt(walking(fomdgt3_7),543).
orientation(fomdgt3_2,110,544). coords(fomdgt3_2,238,93,544).  visible(fomdgt3_2,544). happensAt(walking(fomdgt3_2),544). orientation(fomdgt3_7,171,544). coords(fomdgt3_7,177,215,544).  visible(fomdgt3_7,544). happensAt(walking(fomdgt3_7),544).
orientation(fomdgt3_2,110,545). coords(fomdgt3_2,239,92,545).  visible(fomdgt3_2,545). happensAt(walking(fomdgt3_2),545). orientation(fomdgt3_7,171,545). coords(fomdgt3_7,177,216,545).  visible(fomdgt3_7,545). happensAt(walking(fomdgt3_7),545).
orientation(fomdgt3_2,110,546). coords(fomdgt3_2,241,91,546).  visible(fomdgt3_2,546). happensAt(walking(fomdgt3_2),546). orientation(fomdgt3_7,171,546). coords(fomdgt3_7,183,218,546).  visible(fomdgt3_7,546). happensAt(walking(fomdgt3_7),546).
orientation(fomdgt3_2,110,547). coords(fomdgt3_2,243,90,547).  visible(fomdgt3_2,547). happensAt(walking(fomdgt3_2),547). orientation(fomdgt3_7,171,547). coords(fomdgt3_7,185,219,547).  visible(fomdgt3_7,547). happensAt(walking(fomdgt3_7),547).
orientation(fomdgt3_2,110,548). coords(fomdgt3_2,243,90,548).  visible(fomdgt3_2,548). happensAt(walking(fomdgt3_2),548). orientation(fomdgt3_7,171,548). coords(fomdgt3_7,187,220,548).  visible(fomdgt3_7,548). happensAt(walking(fomdgt3_7),548).
orientation(fomdgt3_2,110,549). coords(fomdgt3_2,245,90,549).  visible(fomdgt3_2,549). happensAt(walking(fomdgt3_2),549). orientation(fomdgt3_7,171,549). coords(fomdgt3_7,189,221,549).  visible(fomdgt3_7,549). happensAt(walking(fomdgt3_7),549).
orientation(fomdgt3_2,110,550). coords(fomdgt3_2,245,90,550).  visible(fomdgt3_2,550). happensAt(walking(fomdgt3_2),550). orientation(fomdgt3_7,171,550). coords(fomdgt3_7,191,221,550).  visible(fomdgt3_7,550). happensAt(walking(fomdgt3_7),550).
orientation(fomdgt3_2,110,551). coords(fomdgt3_2,246,91,551).  visible(fomdgt3_2,551). happensAt(walking(fomdgt3_2),551). orientation(fomdgt3_7,171,551). coords(fomdgt3_7,193,221,551).  visible(fomdgt3_7,551). happensAt(walking(fomdgt3_7),551).
orientation(fomdgt3_2,110,552). coords(fomdgt3_2,247,91,552).  visible(fomdgt3_2,552). happensAt(walking(fomdgt3_2),552). orientation(fomdgt3_7,171,552). coords(fomdgt3_7,192,224,552).  visible(fomdgt3_7,552). happensAt(walking(fomdgt3_7),552).
orientation(fomdgt3_2,110,553). coords(fomdgt3_2,248,91,553).  visible(fomdgt3_2,553). happensAt(walking(fomdgt3_2),553). orientation(fomdgt3_7,171,553). coords(fomdgt3_7,194,224,553).  visible(fomdgt3_7,553). happensAt(walking(fomdgt3_7),553).
orientation(fomdgt3_2,110,554). coords(fomdgt3_2,249,91,554).  visible(fomdgt3_2,554). happensAt(walking(fomdgt3_2),554). orientation(fomdgt3_7,171,554). coords(fomdgt3_7,195,224,554).  visible(fomdgt3_7,554). happensAt(walking(fomdgt3_7),554).
orientation(fomdgt3_2,110,555). coords(fomdgt3_2,249,91,555).  visible(fomdgt3_2,555). happensAt(walking(fomdgt3_2),555). orientation(fomdgt3_7,171,555). coords(fomdgt3_7,195,224,555).  visible(fomdgt3_7,555). happensAt(walking(fomdgt3_7),555).
orientation(fomdgt3_2,110,556). coords(fomdgt3_2,249,91,556).  visible(fomdgt3_2,556). happensAt(walking(fomdgt3_2),556). orientation(fomdgt3_7,171,556). coords(fomdgt3_7,196,224,556).  visible(fomdgt3_7,556). happensAt(walking(fomdgt3_7),556).
orientation(fomdgt3_2,110,557). coords(fomdgt3_2,249,91,557).  visible(fomdgt3_2,557). happensAt(walking(fomdgt3_2),557). orientation(fomdgt3_7,171,557). coords(fomdgt3_7,196,227,557).  visible(fomdgt3_7,557). happensAt(walking(fomdgt3_7),557).
orientation(fomdgt3_2,110,558). coords(fomdgt3_2,249,91,558).  visible(fomdgt3_2,558). happensAt(walking(fomdgt3_2),558). orientation(fomdgt3_7,171,558). coords(fomdgt3_7,197,227,558).  visible(fomdgt3_7,558). happensAt(walking(fomdgt3_7),558).
orientation(fomdgt3_2,110,559). coords(fomdgt3_2,249,91,559).  visible(fomdgt3_2,559). happensAt(walking(fomdgt3_2),559). orientation(fomdgt3_7,171,559). coords(fomdgt3_7,198,228,559).  visible(fomdgt3_7,559). happensAt(walking(fomdgt3_7),559).
orientation(fomdgt3_2,110,560). coords(fomdgt3_2,249,91,560).  visible(fomdgt3_2,560). happensAt(walking(fomdgt3_2),560). orientation(fomdgt3_7,171,560). coords(fomdgt3_7,199,229,560).  visible(fomdgt3_7,560). happensAt(walking(fomdgt3_7),560).
orientation(fomdgt3_2,130,561). coords(fomdgt3_2,250,91,561).  visible(fomdgt3_2,561). happensAt(walking(fomdgt3_2),561). orientation(fomdgt3_7,171,561). coords(fomdgt3_7,202,232,561).  visible(fomdgt3_7,561). happensAt(walking(fomdgt3_7),561).
orientation(fomdgt3_2,130,562). coords(fomdgt3_2,250,91,562).  visible(fomdgt3_2,562). happensAt(walking(fomdgt3_2),562). orientation(fomdgt3_7,171,562). coords(fomdgt3_7,204,236,562).  visible(fomdgt3_7,562). happensAt(walking(fomdgt3_7),562).
orientation(fomdgt3_2,130,563). coords(fomdgt3_2,250,91,563).  visible(fomdgt3_2,563). happensAt(walking(fomdgt3_2),563). orientation(fomdgt3_7,171,563). coords(fomdgt3_7,206,237,563).  visible(fomdgt3_7,563). happensAt(walking(fomdgt3_7),563).
orientation(fomdgt3_2,130,564). coords(fomdgt3_2,250,91,564).  visible(fomdgt3_2,564). happensAt(walking(fomdgt3_2),564). orientation(fomdgt3_7,171,564). coords(fomdgt3_7,207,238,564).  visible(fomdgt3_7,564). happensAt(walking(fomdgt3_7),564).
orientation(fomdgt3_2,130,565). coords(fomdgt3_2,250,91,565).  visible(fomdgt3_2,565). happensAt(walking(fomdgt3_2),565). orientation(fomdgt3_7,171,565). coords(fomdgt3_7,209,238,565).  visible(fomdgt3_7,565). happensAt(walking(fomdgt3_7),565).
orientation(fomdgt3_2,130,566). coords(fomdgt3_2,250,91,566).  visible(fomdgt3_2,566). happensAt(walking(fomdgt3_2),566). orientation(fomdgt3_7,171,566). coords(fomdgt3_7,210,238,566).  visible(fomdgt3_7,566). happensAt(walking(fomdgt3_7),566).
#show holdsAt/2.