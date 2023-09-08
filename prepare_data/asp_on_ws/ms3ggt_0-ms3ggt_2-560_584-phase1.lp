
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

	orientation(ms3ggt_0,168,560). coords(ms3ggt_0,73,191,560).  visible(ms3ggt_0,560). happensAt(inactive(ms3ggt_0),560). orientation(ms3ggt_2,150,560). coords(ms3ggt_2,84,92,560).  visible(ms3ggt_2,560). happensAt(walking(ms3ggt_2),560).
orientation(ms3ggt_0,168,561). coords(ms3ggt_0,73,191,561).  visible(ms3ggt_0,561). happensAt(inactive(ms3ggt_0),561). orientation(ms3ggt_2,150,561). coords(ms3ggt_2,83,91,561).  visible(ms3ggt_2,561). happensAt(walking(ms3ggt_2),561).
orientation(ms3ggt_0,168,562). coords(ms3ggt_0,73,191,562).  visible(ms3ggt_0,562). happensAt(inactive(ms3ggt_0),562). orientation(ms3ggt_2,150,562). coords(ms3ggt_2,83,90,562).  visible(ms3ggt_2,562). happensAt(walking(ms3ggt_2),562).
orientation(ms3ggt_0,168,563). coords(ms3ggt_0,73,191,563).  visible(ms3ggt_0,563). happensAt(inactive(ms3ggt_0),563). orientation(ms3ggt_2,150,563). coords(ms3ggt_2,83,90,563).  visible(ms3ggt_2,563). happensAt(walking(ms3ggt_2),563).
orientation(ms3ggt_0,168,564). coords(ms3ggt_0,73,191,564).  visible(ms3ggt_0,564). happensAt(inactive(ms3ggt_0),564). orientation(ms3ggt_2,150,564). coords(ms3ggt_2,83,90,564).  visible(ms3ggt_2,564). happensAt(walking(ms3ggt_2),564).
orientation(ms3ggt_0,168,565). coords(ms3ggt_0,73,191,565).  visible(ms3ggt_0,565). happensAt(inactive(ms3ggt_0),565). orientation(ms3ggt_2,150,565). coords(ms3ggt_2,83,90,565).  visible(ms3ggt_2,565). happensAt(walking(ms3ggt_2),565).
orientation(ms3ggt_0,168,566). coords(ms3ggt_0,73,191,566).  visible(ms3ggt_0,566). happensAt(inactive(ms3ggt_0),566). orientation(ms3ggt_2,150,566). coords(ms3ggt_2,83,90,566).  visible(ms3ggt_2,566). happensAt(walking(ms3ggt_2),566).
orientation(ms3ggt_0,168,567). coords(ms3ggt_0,73,191,567).  visible(ms3ggt_0,567). happensAt(inactive(ms3ggt_0),567). orientation(ms3ggt_2,150,567). coords(ms3ggt_2,83,90,567).  visible(ms3ggt_2,567). happensAt(walking(ms3ggt_2),567).
orientation(ms3ggt_0,168,568). coords(ms3ggt_0,73,191,568).  visible(ms3ggt_0,568). happensAt(inactive(ms3ggt_0),568). orientation(ms3ggt_2,150,568). coords(ms3ggt_2,83,90,568).  visible(ms3ggt_2,568). happensAt(walking(ms3ggt_2),568).
orientation(ms3ggt_0,168,569). coords(ms3ggt_0,73,192,569).  visible(ms3ggt_0,569). happensAt(inactive(ms3ggt_0),569). orientation(ms3ggt_2,150,569). coords(ms3ggt_2,83,89,569).  visible(ms3ggt_2,569). happensAt(walking(ms3ggt_2),569).
orientation(ms3ggt_0,168,570). coords(ms3ggt_0,73,192,570).  visible(ms3ggt_0,570). happensAt(inactive(ms3ggt_0),570). orientation(ms3ggt_2,150,570). coords(ms3ggt_2,83,89,570).  visible(ms3ggt_2,570). happensAt(walking(ms3ggt_2),570).
orientation(ms3ggt_0,168,571). coords(ms3ggt_0,73,192,571).  visible(ms3ggt_0,571). happensAt(inactive(ms3ggt_0),571). orientation(ms3ggt_2,150,571). coords(ms3ggt_2,83,89,571).  visible(ms3ggt_2,571). happensAt(walking(ms3ggt_2),571).
orientation(ms3ggt_0,168,572). coords(ms3ggt_0,73,192,572).  visible(ms3ggt_0,572). happensAt(inactive(ms3ggt_0),572). orientation(ms3ggt_2,150,572). coords(ms3ggt_2,83,89,572).  visible(ms3ggt_2,572). happensAt(walking(ms3ggt_2),572).
orientation(ms3ggt_0,168,573). coords(ms3ggt_0,73,192,573).  visible(ms3ggt_0,573). happensAt(inactive(ms3ggt_0),573). orientation(ms3ggt_2,150,573). coords(ms3ggt_2,83,89,573).  visible(ms3ggt_2,573). happensAt(walking(ms3ggt_2),573).
orientation(ms3ggt_0,168,574). coords(ms3ggt_0,73,192,574).  visible(ms3ggt_0,574). happensAt(inactive(ms3ggt_0),574). orientation(ms3ggt_2,150,574). coords(ms3ggt_2,83,89,574).  visible(ms3ggt_2,574). happensAt(walking(ms3ggt_2),574).
orientation(ms3ggt_0,168,575). coords(ms3ggt_0,73,192,575).  visible(ms3ggt_0,575). happensAt(inactive(ms3ggt_0),575). orientation(ms3ggt_2,150,575). coords(ms3ggt_2,83,87,575).  visible(ms3ggt_2,575). happensAt(walking(ms3ggt_2),575).
orientation(ms3ggt_0,168,576). coords(ms3ggt_0,73,192,576).  visible(ms3ggt_0,576). happensAt(inactive(ms3ggt_0),576). orientation(ms3ggt_2,150,576). coords(ms3ggt_2,83,86,576).  visible(ms3ggt_2,576). happensAt(walking(ms3ggt_2),576).
orientation(ms3ggt_0,168,577). coords(ms3ggt_0,73,192,577).  visible(ms3ggt_0,577). happensAt(inactive(ms3ggt_0),577). orientation(ms3ggt_2,150,577). coords(ms3ggt_2,83,86,577).  visible(ms3ggt_2,577). happensAt(walking(ms3ggt_2),577).
orientation(ms3ggt_0,168,578). coords(ms3ggt_0,73,192,578).  visible(ms3ggt_0,578). happensAt(inactive(ms3ggt_0),578). orientation(ms3ggt_2,150,578). coords(ms3ggt_2,83,85,578).  visible(ms3ggt_2,578). happensAt(walking(ms3ggt_2),578).
orientation(ms3ggt_0,168,579). coords(ms3ggt_0,73,192,579).  visible(ms3ggt_0,579). happensAt(inactive(ms3ggt_0),579). orientation(ms3ggt_2,150,579). coords(ms3ggt_2,83,85,579).  visible(ms3ggt_2,579). happensAt(walking(ms3ggt_2),579).
orientation(ms3ggt_0,168,580). coords(ms3ggt_0,73,192,580).  visible(ms3ggt_0,580). happensAt(inactive(ms3ggt_0),580). orientation(ms3ggt_2,150,580). coords(ms3ggt_2,83,85,580).  visible(ms3ggt_2,580). happensAt(walking(ms3ggt_2),580).
orientation(ms3ggt_0,168,581). coords(ms3ggt_0,73,192,581).  visible(ms3ggt_0,581). happensAt(inactive(ms3ggt_0),581). orientation(ms3ggt_2,150,581). coords(ms3ggt_2,83,85,581).  visible(ms3ggt_2,581). happensAt(walking(ms3ggt_2),581).
orientation(ms3ggt_0,168,582). coords(ms3ggt_0,73,192,582).  visible(ms3ggt_0,582). happensAt(inactive(ms3ggt_0),582). orientation(ms3ggt_2,150,582). coords(ms3ggt_2,83,84,582).  visible(ms3ggt_2,582). happensAt(walking(ms3ggt_2),582).
orientation(ms3ggt_0,168,583). coords(ms3ggt_0,73,192,583).  visible(ms3ggt_0,583). happensAt(inactive(ms3ggt_0),583). orientation(ms3ggt_2,150,583). coords(ms3ggt_2,83,84,583).  visible(ms3ggt_2,583). happensAt(walking(ms3ggt_2),583).
orientation(ms3ggt_0,168,584). coords(ms3ggt_0,73,192,584).  visible(ms3ggt_0,584). happensAt(inactive(ms3ggt_0),584). orientation(ms3ggt_2,150,584). coords(ms3ggt_2,83,83,584).  visible(ms3ggt_2,584). happensAt(walking(ms3ggt_2),584).
#show holdsAt/2.