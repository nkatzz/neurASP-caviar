
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

	orientation(bww1gt_1,173,554). coords(bww1gt_1,99,249,554).  visible(bww1gt_1,554). happensAt(inactive(bww1gt_1),554). orientation(bww1gt_2,169,554). coords(bww1gt_2,160,206,554).  visible(bww1gt_2,554). happensAt(walking(bww1gt_2),554).
orientation(bww1gt_1,173,555). coords(bww1gt_1,99,249,555).  visible(bww1gt_1,555). happensAt(inactive(bww1gt_1),555). orientation(bww1gt_2,169,555). coords(bww1gt_2,158,206,555).  visible(bww1gt_2,555). happensAt(walking(bww1gt_2),555).
orientation(bww1gt_1,173,556). coords(bww1gt_1,99,249,556).  visible(bww1gt_1,556). happensAt(inactive(bww1gt_1),556). orientation(bww1gt_2,169,556). coords(bww1gt_2,158,205,556).  visible(bww1gt_2,556). happensAt(walking(bww1gt_2),556).
orientation(bww1gt_1,173,557). coords(bww1gt_1,99,249,557).  visible(bww1gt_1,557). happensAt(inactive(bww1gt_1),557). orientation(bww1gt_2,169,557). coords(bww1gt_2,155,205,557).  visible(bww1gt_2,557). happensAt(walking(bww1gt_2),557).
orientation(bww1gt_1,173,558). coords(bww1gt_1,99,249,558).  visible(bww1gt_1,558). happensAt(inactive(bww1gt_1),558). orientation(bww1gt_2,169,558). coords(bww1gt_2,154,205,558).  visible(bww1gt_2,558). happensAt(walking(bww1gt_2),558).
orientation(bww1gt_1,173,559). coords(bww1gt_1,99,249,559).  visible(bww1gt_1,559). happensAt(inactive(bww1gt_1),559). orientation(bww1gt_2,169,559). coords(bww1gt_2,153,205,559).  visible(bww1gt_2,559). happensAt(walking(bww1gt_2),559).
orientation(bww1gt_1,173,560). coords(bww1gt_1,99,249,560).  visible(bww1gt_1,560). happensAt(inactive(bww1gt_1),560). orientation(bww1gt_2,169,560). coords(bww1gt_2,153,205,560).  visible(bww1gt_2,560). happensAt(walking(bww1gt_2),560).
orientation(bww1gt_1,173,561). coords(bww1gt_1,99,249,561).  visible(bww1gt_1,561). happensAt(inactive(bww1gt_1),561). orientation(bww1gt_2,169,561). coords(bww1gt_2,152,205,561).  visible(bww1gt_2,561). happensAt(walking(bww1gt_2),561).
orientation(bww1gt_1,173,562). coords(bww1gt_1,99,249,562).  visible(bww1gt_1,562). happensAt(inactive(bww1gt_1),562). orientation(bww1gt_2,169,562). coords(bww1gt_2,152,206,562).  visible(bww1gt_2,562). happensAt(walking(bww1gt_2),562).
orientation(bww1gt_1,173,563). coords(bww1gt_1,99,248,563).  visible(bww1gt_1,563). happensAt(inactive(bww1gt_1),563). orientation(bww1gt_2,169,563). coords(bww1gt_2,152,206,563).  visible(bww1gt_2,563). happensAt(walking(bww1gt_2),563).
orientation(bww1gt_1,173,564). coords(bww1gt_1,99,248,564).  visible(bww1gt_1,564). happensAt(inactive(bww1gt_1),564). orientation(bww1gt_2,169,564). coords(bww1gt_2,152,206,564).  visible(bww1gt_2,564). happensAt(walking(bww1gt_2),564).
orientation(bww1gt_1,173,565). coords(bww1gt_1,99,248,565).  visible(bww1gt_1,565). happensAt(inactive(bww1gt_1),565). orientation(bww1gt_2,169,565). coords(bww1gt_2,151,206,565).  visible(bww1gt_2,565). happensAt(walking(bww1gt_2),565).
orientation(bww1gt_1,173,566). coords(bww1gt_1,99,248,566).  visible(bww1gt_1,566). happensAt(inactive(bww1gt_1),566). orientation(bww1gt_2,169,566). coords(bww1gt_2,151,206,566).  visible(bww1gt_2,566). happensAt(walking(bww1gt_2),566).
orientation(bww1gt_1,173,567). coords(bww1gt_1,99,248,567).  visible(bww1gt_1,567). happensAt(inactive(bww1gt_1),567). orientation(bww1gt_2,169,567). coords(bww1gt_2,151,206,567).  visible(bww1gt_2,567). happensAt(walking(bww1gt_2),567).
orientation(bww1gt_1,173,568). coords(bww1gt_1,99,248,568).  visible(bww1gt_1,568). happensAt(inactive(bww1gt_1),568). orientation(bww1gt_2,169,568). coords(bww1gt_2,150,206,568).  visible(bww1gt_2,568). happensAt(walking(bww1gt_2),568).
orientation(bww1gt_1,173,569). coords(bww1gt_1,99,248,569).  visible(bww1gt_1,569). happensAt(inactive(bww1gt_1),569). orientation(bww1gt_2,169,569). coords(bww1gt_2,150,207,569).  visible(bww1gt_2,569). happensAt(walking(bww1gt_2),569).
orientation(bww1gt_1,173,570). coords(bww1gt_1,99,248,570).  visible(bww1gt_1,570). happensAt(inactive(bww1gt_1),570). orientation(bww1gt_2,169,570). coords(bww1gt_2,149,208,570).  visible(bww1gt_2,570). happensAt(walking(bww1gt_2),570).
orientation(bww1gt_1,173,571). coords(bww1gt_1,99,248,571).  visible(bww1gt_1,571). happensAt(inactive(bww1gt_1),571). orientation(bww1gt_2,169,571). coords(bww1gt_2,148,209,571).  visible(bww1gt_2,571). happensAt(walking(bww1gt_2),571).
orientation(bww1gt_1,173,572). coords(bww1gt_1,99,249,572).  visible(bww1gt_1,572). happensAt(inactive(bww1gt_1),572). orientation(bww1gt_2,169,572). coords(bww1gt_2,147,209,572).  visible(bww1gt_2,572). happensAt(walking(bww1gt_2),572).
orientation(bww1gt_1,173,573). coords(bww1gt_1,99,249,573).  visible(bww1gt_1,573). happensAt(inactive(bww1gt_1),573). orientation(bww1gt_2,169,573). coords(bww1gt_2,145,209,573).  visible(bww1gt_2,573). happensAt(walking(bww1gt_2),573).
orientation(bww1gt_1,173,574). coords(bww1gt_1,99,249,574).  visible(bww1gt_1,574). happensAt(inactive(bww1gt_1),574). orientation(bww1gt_2,169,574). coords(bww1gt_2,144,209,574).  visible(bww1gt_2,574). happensAt(walking(bww1gt_2),574).
orientation(bww1gt_1,173,575). coords(bww1gt_1,99,249,575).  visible(bww1gt_1,575). happensAt(inactive(bww1gt_1),575). orientation(bww1gt_2,169,575). coords(bww1gt_2,143,209,575).  visible(bww1gt_2,575). happensAt(walking(bww1gt_2),575).
orientation(bww1gt_1,173,576). coords(bww1gt_1,99,249,576).  visible(bww1gt_1,576). happensAt(inactive(bww1gt_1),576). orientation(bww1gt_2,169,576). coords(bww1gt_2,141,210,576).  visible(bww1gt_2,576). happensAt(walking(bww1gt_2),576).
orientation(bww1gt_1,173,577). coords(bww1gt_1,99,249,577).  visible(bww1gt_1,577). happensAt(inactive(bww1gt_1),577). orientation(bww1gt_2,169,577). coords(bww1gt_2,141,210,577).  visible(bww1gt_2,577). happensAt(walking(bww1gt_2),577).
orientation(bww1gt_1,173,578). coords(bww1gt_1,99,249,578).  visible(bww1gt_1,578). happensAt(inactive(bww1gt_1),578). orientation(bww1gt_2,176,578). coords(bww1gt_2,139,210,578).  visible(bww1gt_2,578). happensAt(walking(bww1gt_2),578).
#show holdsAt/2.