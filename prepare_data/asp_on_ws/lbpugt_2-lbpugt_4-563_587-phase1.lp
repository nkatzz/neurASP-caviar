
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

	orientation(lbpugt_2,132,563). coords(lbpugt_2,180,49,563).  visible(lbpugt_2,563). happensAt(active(lbpugt_2),563). orientation(lbpugt_4,135,563). coords(lbpugt_4,219,34,563).  visible(lbpugt_4,563). happensAt(inactive(lbpugt_4),563).
orientation(lbpugt_2,132,564). coords(lbpugt_2,181,49,564).  visible(lbpugt_2,564). happensAt(inactive(lbpugt_2),564). orientation(lbpugt_4,135,564). coords(lbpugt_4,219,34,564).  visible(lbpugt_4,564). happensAt(inactive(lbpugt_4),564).
orientation(lbpugt_2,132,565). coords(lbpugt_2,181,49,565).  visible(lbpugt_2,565). happensAt(inactive(lbpugt_2),565). orientation(lbpugt_4,135,565). coords(lbpugt_4,219,34,565).  visible(lbpugt_4,565). happensAt(inactive(lbpugt_4),565).
orientation(lbpugt_2,132,566). coords(lbpugt_2,181,49,566).  visible(lbpugt_2,566). happensAt(inactive(lbpugt_2),566). orientation(lbpugt_4,135,566). coords(lbpugt_4,219,34,566).  visible(lbpugt_4,566). happensAt(inactive(lbpugt_4),566).
orientation(lbpugt_2,132,567). coords(lbpugt_2,181,49,567).  visible(lbpugt_2,567). happensAt(inactive(lbpugt_2),567). orientation(lbpugt_4,135,567). coords(lbpugt_4,219,34,567).  visible(lbpugt_4,567). happensAt(inactive(lbpugt_4),567).
orientation(lbpugt_2,132,568). coords(lbpugt_2,181,49,568).  visible(lbpugt_2,568). happensAt(inactive(lbpugt_2),568). orientation(lbpugt_4,135,568). coords(lbpugt_4,219,34,568).  visible(lbpugt_4,568). happensAt(inactive(lbpugt_4),568).
orientation(lbpugt_2,132,569). coords(lbpugt_2,181,49,569).  visible(lbpugt_2,569). happensAt(inactive(lbpugt_2),569). orientation(lbpugt_4,135,569). coords(lbpugt_4,219,34,569).  visible(lbpugt_4,569). happensAt(inactive(lbpugt_4),569).
orientation(lbpugt_2,132,570). coords(lbpugt_2,181,49,570).  visible(lbpugt_2,570). happensAt(inactive(lbpugt_2),570). orientation(lbpugt_4,135,570). coords(lbpugt_4,219,34,570).  visible(lbpugt_4,570). happensAt(inactive(lbpugt_4),570).
orientation(lbpugt_2,132,571). coords(lbpugt_2,181,49,571).  visible(lbpugt_2,571). happensAt(inactive(lbpugt_2),571). orientation(lbpugt_4,135,571). coords(lbpugt_4,219,34,571).  visible(lbpugt_4,571). happensAt(inactive(lbpugt_4),571).
orientation(lbpugt_2,132,572). coords(lbpugt_2,181,49,572).  visible(lbpugt_2,572). happensAt(inactive(lbpugt_2),572). orientation(lbpugt_4,135,572). coords(lbpugt_4,219,34,572).  visible(lbpugt_4,572). happensAt(inactive(lbpugt_4),572).
orientation(lbpugt_2,132,573). coords(lbpugt_2,181,49,573).  visible(lbpugt_2,573). happensAt(inactive(lbpugt_2),573). orientation(lbpugt_4,135,573). coords(lbpugt_4,219,34,573).  visible(lbpugt_4,573). happensAt(inactive(lbpugt_4),573).
orientation(lbpugt_2,132,574). coords(lbpugt_2,181,49,574).  visible(lbpugt_2,574). happensAt(inactive(lbpugt_2),574). orientation(lbpugt_4,135,574). coords(lbpugt_4,219,34,574).  visible(lbpugt_4,574). happensAt(inactive(lbpugt_4),574).
orientation(lbpugt_2,132,575). coords(lbpugt_2,181,50,575).  visible(lbpugt_2,575). happensAt(inactive(lbpugt_2),575). orientation(lbpugt_4,135,575). coords(lbpugt_4,219,34,575).  visible(lbpugt_4,575). happensAt(inactive(lbpugt_4),575).
orientation(lbpugt_2,132,576). coords(lbpugt_2,181,50,576).  visible(lbpugt_2,576). happensAt(inactive(lbpugt_2),576). orientation(lbpugt_4,135,576). coords(lbpugt_4,219,34,576).  visible(lbpugt_4,576). happensAt(inactive(lbpugt_4),576).
orientation(lbpugt_2,132,577). coords(lbpugt_2,181,50,577).  visible(lbpugt_2,577). happensAt(inactive(lbpugt_2),577). orientation(lbpugt_4,135,577). coords(lbpugt_4,219,34,577).  visible(lbpugt_4,577). happensAt(inactive(lbpugt_4),577).
orientation(lbpugt_2,132,578). coords(lbpugt_2,181,50,578).  visible(lbpugt_2,578). happensAt(inactive(lbpugt_2),578). orientation(lbpugt_4,135,578). coords(lbpugt_4,219,34,578).  visible(lbpugt_4,578). happensAt(inactive(lbpugt_4),578).
orientation(lbpugt_2,132,579). coords(lbpugt_2,181,50,579).  visible(lbpugt_2,579). happensAt(inactive(lbpugt_2),579). orientation(lbpugt_4,135,579). coords(lbpugt_4,219,34,579).  visible(lbpugt_4,579). happensAt(inactive(lbpugt_4),579).
orientation(lbpugt_2,132,580). coords(lbpugt_2,181,50,580).  visible(lbpugt_2,580). happensAt(inactive(lbpugt_2),580). orientation(lbpugt_4,135,580). coords(lbpugt_4,219,34,580).  visible(lbpugt_4,580). happensAt(inactive(lbpugt_4),580).
orientation(lbpugt_2,132,581). coords(lbpugt_2,181,50,581).  visible(lbpugt_2,581). happensAt(inactive(lbpugt_2),581). orientation(lbpugt_4,135,581). coords(lbpugt_4,219,34,581).  visible(lbpugt_4,581). happensAt(inactive(lbpugt_4),581).
orientation(lbpugt_2,132,582). coords(lbpugt_2,181,50,582).  visible(lbpugt_2,582). happensAt(inactive(lbpugt_2),582). orientation(lbpugt_4,135,582). coords(lbpugt_4,219,34,582).  visible(lbpugt_4,582). happensAt(inactive(lbpugt_4),582).
orientation(lbpugt_2,132,583). coords(lbpugt_2,181,50,583).  visible(lbpugt_2,583). happensAt(inactive(lbpugt_2),583). orientation(lbpugt_4,135,583). coords(lbpugt_4,219,34,583).  visible(lbpugt_4,583). happensAt(inactive(lbpugt_4),583).
orientation(lbpugt_2,132,584). coords(lbpugt_2,181,50,584).  visible(lbpugt_2,584). happensAt(inactive(lbpugt_2),584). orientation(lbpugt_4,135,584). coords(lbpugt_4,219,34,584).  visible(lbpugt_4,584). happensAt(inactive(lbpugt_4),584).
orientation(lbpugt_2,132,585). coords(lbpugt_2,181,50,585).  visible(lbpugt_2,585). happensAt(inactive(lbpugt_2),585). orientation(lbpugt_4,135,585). coords(lbpugt_4,219,34,585).  visible(lbpugt_4,585). happensAt(inactive(lbpugt_4),585).
orientation(lbpugt_2,132,586). coords(lbpugt_2,181,50,586).  visible(lbpugt_2,586). happensAt(inactive(lbpugt_2),586). orientation(lbpugt_4,135,586). coords(lbpugt_4,219,34,586).  visible(lbpugt_4,586). happensAt(inactive(lbpugt_4),586).
orientation(lbpugt_2,132,587). coords(lbpugt_2,181,50,587).  visible(lbpugt_2,587). happensAt(inactive(lbpugt_2),587). orientation(lbpugt_4,135,587). coords(lbpugt_4,219,34,587).  visible(lbpugt_4,587). happensAt(inactive(lbpugt_4),587).
#show holdsAt/2.