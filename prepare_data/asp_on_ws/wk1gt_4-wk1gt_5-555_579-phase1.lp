
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

	orientation(wk1gt_4,157,555). coords(wk1gt_4,56,170,555).  visible(wk1gt_4,555). happensAt(inactive(wk1gt_4),555). orientation(wk1gt_5,162,555). coords(wk1gt_5,57,150,555).  visible(wk1gt_5,555). happensAt(inactive(wk1gt_5),555).
orientation(wk1gt_4,157,556). coords(wk1gt_4,56,170,556).  visible(wk1gt_4,556). happensAt(inactive(wk1gt_4),556). orientation(wk1gt_5,162,556). coords(wk1gt_5,57,150,556).  visible(wk1gt_5,556). happensAt(inactive(wk1gt_5),556).
orientation(wk1gt_4,157,557). coords(wk1gt_4,56,170,557).  visible(wk1gt_4,557). happensAt(inactive(wk1gt_4),557). orientation(wk1gt_5,162,557). coords(wk1gt_5,57,150,557).  visible(wk1gt_5,557). happensAt(inactive(wk1gt_5),557).
orientation(wk1gt_4,157,558). coords(wk1gt_4,56,170,558).  visible(wk1gt_4,558). happensAt(inactive(wk1gt_4),558). orientation(wk1gt_5,162,558). coords(wk1gt_5,57,150,558).  visible(wk1gt_5,558). happensAt(inactive(wk1gt_5),558).
orientation(wk1gt_4,157,559). coords(wk1gt_4,56,170,559).  visible(wk1gt_4,559). happensAt(inactive(wk1gt_4),559). orientation(wk1gt_5,162,559). coords(wk1gt_5,57,150,559).  visible(wk1gt_5,559). happensAt(inactive(wk1gt_5),559).
orientation(wk1gt_4,157,560). coords(wk1gt_4,56,170,560).  visible(wk1gt_4,560). happensAt(inactive(wk1gt_4),560). orientation(wk1gt_5,162,560). coords(wk1gt_5,57,150,560).  visible(wk1gt_5,560). happensAt(inactive(wk1gt_5),560).
orientation(wk1gt_4,157,561). coords(wk1gt_4,56,170,561).  visible(wk1gt_4,561). happensAt(inactive(wk1gt_4),561). orientation(wk1gt_5,162,561). coords(wk1gt_5,57,150,561).  visible(wk1gt_5,561). happensAt(inactive(wk1gt_5),561).
orientation(wk1gt_4,157,562). coords(wk1gt_4,56,170,562).  visible(wk1gt_4,562). happensAt(inactive(wk1gt_4),562). orientation(wk1gt_5,162,562). coords(wk1gt_5,57,150,562).  visible(wk1gt_5,562). happensAt(inactive(wk1gt_5),562).
orientation(wk1gt_4,157,563). coords(wk1gt_4,56,170,563).  visible(wk1gt_4,563). happensAt(inactive(wk1gt_4),563). orientation(wk1gt_5,162,563). coords(wk1gt_5,57,150,563).  visible(wk1gt_5,563). happensAt(inactive(wk1gt_5),563).
orientation(wk1gt_4,157,564). coords(wk1gt_4,56,170,564).  visible(wk1gt_4,564). happensAt(inactive(wk1gt_4),564). orientation(wk1gt_5,162,564). coords(wk1gt_5,57,150,564).  visible(wk1gt_5,564). happensAt(inactive(wk1gt_5),564).
orientation(wk1gt_4,157,565). coords(wk1gt_4,56,170,565).  visible(wk1gt_4,565). happensAt(inactive(wk1gt_4),565). orientation(wk1gt_5,162,565). coords(wk1gt_5,57,150,565).  visible(wk1gt_5,565). happensAt(inactive(wk1gt_5),565).
orientation(wk1gt_4,157,566). coords(wk1gt_4,56,170,566).  visible(wk1gt_4,566). happensAt(inactive(wk1gt_4),566). orientation(wk1gt_5,162,566). coords(wk1gt_5,57,150,566).  visible(wk1gt_5,566). happensAt(inactive(wk1gt_5),566).
orientation(wk1gt_4,157,567). coords(wk1gt_4,56,170,567).  visible(wk1gt_4,567). happensAt(inactive(wk1gt_4),567). orientation(wk1gt_5,162,567). coords(wk1gt_5,57,150,567).  visible(wk1gt_5,567). happensAt(inactive(wk1gt_5),567).
orientation(wk1gt_4,157,568). coords(wk1gt_4,56,170,568).  visible(wk1gt_4,568). happensAt(inactive(wk1gt_4),568). orientation(wk1gt_5,162,568). coords(wk1gt_5,57,150,568).  visible(wk1gt_5,568). happensAt(inactive(wk1gt_5),568).
orientation(wk1gt_4,157,569). coords(wk1gt_4,56,170,569).  visible(wk1gt_4,569). happensAt(inactive(wk1gt_4),569). orientation(wk1gt_5,162,569). coords(wk1gt_5,57,150,569).  visible(wk1gt_5,569). happensAt(inactive(wk1gt_5),569).
orientation(wk1gt_4,157,570). coords(wk1gt_4,56,170,570).  visible(wk1gt_4,570). happensAt(inactive(wk1gt_4),570). orientation(wk1gt_5,162,570). coords(wk1gt_5,57,150,570).  visible(wk1gt_5,570). happensAt(inactive(wk1gt_5),570).
orientation(wk1gt_4,157,571). coords(wk1gt_4,56,170,571).  visible(wk1gt_4,571). happensAt(inactive(wk1gt_4),571). orientation(wk1gt_5,162,571). coords(wk1gt_5,57,150,571).  visible(wk1gt_5,571). happensAt(inactive(wk1gt_5),571).
orientation(wk1gt_4,157,572). coords(wk1gt_4,56,170,572).  visible(wk1gt_4,572). happensAt(inactive(wk1gt_4),572). orientation(wk1gt_5,162,572). coords(wk1gt_5,57,150,572).  visible(wk1gt_5,572). happensAt(inactive(wk1gt_5),572).
orientation(wk1gt_4,157,573). coords(wk1gt_4,56,170,573).  visible(wk1gt_4,573). happensAt(inactive(wk1gt_4),573). orientation(wk1gt_5,162,573). coords(wk1gt_5,57,150,573).  visible(wk1gt_5,573). happensAt(inactive(wk1gt_5),573).
orientation(wk1gt_4,157,574). coords(wk1gt_4,56,170,574).  visible(wk1gt_4,574). happensAt(inactive(wk1gt_4),574). orientation(wk1gt_5,162,574). coords(wk1gt_5,57,150,574).  visible(wk1gt_5,574). happensAt(inactive(wk1gt_5),574).
orientation(wk1gt_4,157,575). coords(wk1gt_4,56,170,575).  visible(wk1gt_4,575). happensAt(inactive(wk1gt_4),575). orientation(wk1gt_5,162,575). coords(wk1gt_5,57,150,575).  visible(wk1gt_5,575). happensAt(inactive(wk1gt_5),575).
orientation(wk1gt_4,157,576). coords(wk1gt_4,56,170,576).  visible(wk1gt_4,576). happensAt(inactive(wk1gt_4),576). orientation(wk1gt_5,162,576). coords(wk1gt_5,57,150,576).  visible(wk1gt_5,576). happensAt(inactive(wk1gt_5),576).
orientation(wk1gt_4,157,577). coords(wk1gt_4,56,170,577).  visible(wk1gt_4,577). happensAt(inactive(wk1gt_4),577). orientation(wk1gt_5,162,577). coords(wk1gt_5,57,150,577).  visible(wk1gt_5,577). happensAt(inactive(wk1gt_5),577).
orientation(wk1gt_4,157,578). coords(wk1gt_4,56,170,578).  visible(wk1gt_4,578). happensAt(inactive(wk1gt_4),578). orientation(wk1gt_5,162,578). coords(wk1gt_5,57,150,578).  visible(wk1gt_5,578). happensAt(inactive(wk1gt_5),578).
orientation(wk1gt_4,157,579). coords(wk1gt_4,56,170,579).  visible(wk1gt_4,579). happensAt(inactive(wk1gt_4),579). orientation(wk1gt_5,162,579). coords(wk1gt_5,57,150,579).  visible(wk1gt_5,579). happensAt(inactive(wk1gt_5),579).
#show holdsAt/2.