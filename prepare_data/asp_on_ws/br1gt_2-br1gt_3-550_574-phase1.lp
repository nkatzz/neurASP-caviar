
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
orientation(br1gt_2,12,563). coords(br1gt_2,86,209,563).  visible(br1gt_2,563). happensAt(inactive(br1gt_2),563). orientation(br1gt_3,145,563). coords(br1gt_3,203,108,563).  visible(br1gt_3,563). happensAt(walking(br1gt_3),563).
orientation(br1gt_2,12,564). coords(br1gt_2,86,209,564).  visible(br1gt_2,564). happensAt(inactive(br1gt_2),564). orientation(br1gt_3,145,564). coords(br1gt_3,207,108,564).  visible(br1gt_3,564). happensAt(walking(br1gt_3),564).
orientation(br1gt_2,12,565). coords(br1gt_2,86,209,565).  visible(br1gt_2,565). happensAt(inactive(br1gt_2),565). orientation(br1gt_3,145,565). coords(br1gt_3,211,107,565).  visible(br1gt_3,565). happensAt(walking(br1gt_3),565).
orientation(br1gt_2,12,566). coords(br1gt_2,86,209,566).  visible(br1gt_2,566). happensAt(inactive(br1gt_2),566). orientation(br1gt_3,145,566). coords(br1gt_3,214,107,566).  visible(br1gt_3,566). happensAt(walking(br1gt_3),566).
orientation(br1gt_2,12,567). coords(br1gt_2,86,209,567).  visible(br1gt_2,567). happensAt(inactive(br1gt_2),567). orientation(br1gt_3,145,567). coords(br1gt_3,217,107,567).  visible(br1gt_3,567). happensAt(walking(br1gt_3),567).
orientation(br1gt_2,12,568). coords(br1gt_2,86,209,568).  visible(br1gt_2,568). happensAt(inactive(br1gt_2),568). orientation(br1gt_3,145,568). coords(br1gt_3,219,107,568).  visible(br1gt_3,568). happensAt(walking(br1gt_3),568).
orientation(br1gt_2,12,569). coords(br1gt_2,86,209,569).  visible(br1gt_2,569). happensAt(inactive(br1gt_2),569). orientation(br1gt_3,145,569). coords(br1gt_3,222,106,569).  visible(br1gt_3,569). happensAt(walking(br1gt_3),569).
orientation(br1gt_2,12,570). coords(br1gt_2,86,209,570).  visible(br1gt_2,570). happensAt(inactive(br1gt_2),570). orientation(br1gt_3,145,570). coords(br1gt_3,224,106,570).  visible(br1gt_3,570). happensAt(walking(br1gt_3),570).
orientation(br1gt_2,12,571). coords(br1gt_2,86,209,571).  visible(br1gt_2,571). happensAt(inactive(br1gt_2),571). orientation(br1gt_3,145,571). coords(br1gt_3,224,105,571).  visible(br1gt_3,571). happensAt(walking(br1gt_3),571).
orientation(br1gt_2,12,572). coords(br1gt_2,86,209,572).  visible(br1gt_2,572). happensAt(inactive(br1gt_2),572). orientation(br1gt_3,138,572). coords(br1gt_3,226,105,572).  visible(br1gt_3,572). happensAt(walking(br1gt_3),572).
orientation(br1gt_2,12,573). coords(br1gt_2,86,209,573).  visible(br1gt_2,573). happensAt(inactive(br1gt_2),573). orientation(br1gt_3,138,573). coords(br1gt_3,227,104,573).  visible(br1gt_3,573). happensAt(walking(br1gt_3),573).
orientation(br1gt_2,12,574). coords(br1gt_2,86,209,574).  visible(br1gt_2,574). happensAt(inactive(br1gt_2),574). orientation(br1gt_3,138,574). coords(br1gt_3,228,103,574).  visible(br1gt_3,574). happensAt(walking(br1gt_3),574).
#show holdsAt/2.