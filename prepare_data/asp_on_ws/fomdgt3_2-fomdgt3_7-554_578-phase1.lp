
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
orientation(fomdgt3_2,130,567). coords(fomdgt3_2,249,91,567).  visible(fomdgt3_2,567). happensAt(walking(fomdgt3_2),567). orientation(fomdgt3_7,171,567). coords(fomdgt3_7,211,238,567).  visible(fomdgt3_7,567). happensAt(walking(fomdgt3_7),567).
orientation(fomdgt3_2,130,568). coords(fomdgt3_2,249,91,568).  visible(fomdgt3_2,568). happensAt(walking(fomdgt3_2),568). orientation(fomdgt3_7,171,568). coords(fomdgt3_7,211,238,568).  visible(fomdgt3_7,568). happensAt(walking(fomdgt3_7),568).
orientation(fomdgt3_2,130,569). coords(fomdgt3_2,248,91,569).  visible(fomdgt3_2,569). happensAt(walking(fomdgt3_2),569). orientation(fomdgt3_7,171,569). coords(fomdgt3_7,211,240,569).  visible(fomdgt3_7,569). happensAt(walking(fomdgt3_7),569).
orientation(fomdgt3_2,130,570). coords(fomdgt3_2,248,92,570).  visible(fomdgt3_2,570). happensAt(walking(fomdgt3_2),570). orientation(fomdgt3_7,171,570). coords(fomdgt3_7,212,243,570).  visible(fomdgt3_7,570). happensAt(walking(fomdgt3_7),570).
orientation(fomdgt3_2,130,571). coords(fomdgt3_2,248,92,571).  visible(fomdgt3_2,571). happensAt(walking(fomdgt3_2),571). orientation(fomdgt3_7,171,571). coords(fomdgt3_7,212,243,571).  visible(fomdgt3_7,571). happensAt(walking(fomdgt3_7),571).
orientation(fomdgt3_2,130,572). coords(fomdgt3_2,248,92,572).  visible(fomdgt3_2,572). happensAt(walking(fomdgt3_2),572). orientation(fomdgt3_7,171,572). coords(fomdgt3_7,213,249,572).  visible(fomdgt3_7,572). happensAt(walking(fomdgt3_7),572).
orientation(fomdgt3_2,130,573). coords(fomdgt3_2,247,92,573).  visible(fomdgt3_2,573). happensAt(walking(fomdgt3_2),573). orientation(fomdgt3_7,171,573). coords(fomdgt3_7,214,252,573).  visible(fomdgt3_7,573). happensAt(walking(fomdgt3_7),573).
orientation(fomdgt3_2,130,574). coords(fomdgt3_2,247,92,574).  visible(fomdgt3_2,574). happensAt(walking(fomdgt3_2),574). orientation(fomdgt3_7,171,574). coords(fomdgt3_7,216,256,574).  visible(fomdgt3_7,574). happensAt(walking(fomdgt3_7),574).
orientation(fomdgt3_2,130,575). coords(fomdgt3_2,247,92,575).  visible(fomdgt3_2,575). happensAt(walking(fomdgt3_2),575). orientation(fomdgt3_7,171,575). coords(fomdgt3_7,219,258,575).  visible(fomdgt3_7,575). happensAt(walking(fomdgt3_7),575).
orientation(fomdgt3_2,130,576). coords(fomdgt3_2,247,93,576).  visible(fomdgt3_2,576). happensAt(walking(fomdgt3_2),576). orientation(fomdgt3_7,171,576). coords(fomdgt3_7,222,260,576).  visible(fomdgt3_7,576). happensAt(walking(fomdgt3_7),576).
orientation(fomdgt3_2,130,577). coords(fomdgt3_2,247,94,577).  visible(fomdgt3_2,577). happensAt(walking(fomdgt3_2),577). orientation(fomdgt3_7,171,577). coords(fomdgt3_7,225,261,577).  visible(fomdgt3_7,577). happensAt(walking(fomdgt3_7),577).
orientation(fomdgt3_2,130,578). coords(fomdgt3_2,248,94,578).  visible(fomdgt3_2,578). happensAt(walking(fomdgt3_2),578). orientation(fomdgt3_7,171,578). coords(fomdgt3_7,225,262,578).  visible(fomdgt3_7,578). happensAt(walking(fomdgt3_7),578).
#show holdsAt/2.