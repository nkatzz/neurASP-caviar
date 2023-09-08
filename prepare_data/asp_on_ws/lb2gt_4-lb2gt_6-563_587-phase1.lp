
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

	orientation(lb2gt_4,154,563). coords(lb2gt_4,124,47,563).  visible(lb2gt_4,563). happensAt(walking(lb2gt_4),563). orientation(lb2gt_6,151,563). coords(lb2gt_6,72,122,563).  visible(lb2gt_6,563). happensAt(inactive(lb2gt_6),563).
orientation(lb2gt_4,154,564). coords(lb2gt_4,125,47,564).  visible(lb2gt_4,564). happensAt(walking(lb2gt_4),564). orientation(lb2gt_6,151,564). coords(lb2gt_6,72,122,564).  visible(lb2gt_6,564). happensAt(inactive(lb2gt_6),564).
orientation(lb2gt_4,154,565). coords(lb2gt_4,125,46,565).  visible(lb2gt_4,565). happensAt(walking(lb2gt_4),565). orientation(lb2gt_6,151,565). coords(lb2gt_6,72,122,565).  visible(lb2gt_6,565). happensAt(inactive(lb2gt_6),565).
orientation(lb2gt_4,154,566). coords(lb2gt_4,126,46,566).  visible(lb2gt_4,566). happensAt(walking(lb2gt_4),566). orientation(lb2gt_6,151,566). coords(lb2gt_6,72,122,566).  visible(lb2gt_6,566). happensAt(inactive(lb2gt_6),566).
orientation(lb2gt_4,154,567). coords(lb2gt_4,126,45,567).  visible(lb2gt_4,567). happensAt(walking(lb2gt_4),567). orientation(lb2gt_6,151,567). coords(lb2gt_6,72,122,567).  visible(lb2gt_6,567). happensAt(inactive(lb2gt_6),567).
orientation(lb2gt_4,154,568). coords(lb2gt_4,126,45,568).  visible(lb2gt_4,568). happensAt(walking(lb2gt_4),568). orientation(lb2gt_6,151,568). coords(lb2gt_6,72,122,568).  visible(lb2gt_6,568). happensAt(inactive(lb2gt_6),568).
orientation(lb2gt_4,154,569). coords(lb2gt_4,127,44,569).  visible(lb2gt_4,569). happensAt(walking(lb2gt_4),569). orientation(lb2gt_6,151,569). coords(lb2gt_6,72,122,569).  visible(lb2gt_6,569). happensAt(inactive(lb2gt_6),569).
orientation(lb2gt_4,154,570). coords(lb2gt_4,128,44,570).  visible(lb2gt_4,570). happensAt(walking(lb2gt_4),570). orientation(lb2gt_6,151,570). coords(lb2gt_6,72,122,570).  visible(lb2gt_6,570). happensAt(inactive(lb2gt_6),570).
orientation(lb2gt_4,154,571). coords(lb2gt_4,130,43,571).  visible(lb2gt_4,571). happensAt(walking(lb2gt_4),571). orientation(lb2gt_6,151,571). coords(lb2gt_6,72,122,571).  visible(lb2gt_6,571). happensAt(inactive(lb2gt_6),571).
orientation(lb2gt_4,154,572). coords(lb2gt_4,131,43,572).  visible(lb2gt_4,572). happensAt(walking(lb2gt_4),572). orientation(lb2gt_6,151,572). coords(lb2gt_6,72,122,572).  visible(lb2gt_6,572). happensAt(inactive(lb2gt_6),572).
orientation(lb2gt_4,154,573). coords(lb2gt_4,133,43,573).  visible(lb2gt_4,573). happensAt(walking(lb2gt_4),573). orientation(lb2gt_6,151,573). coords(lb2gt_6,72,122,573).  visible(lb2gt_6,573). happensAt(inactive(lb2gt_6),573).
orientation(lb2gt_4,154,574). coords(lb2gt_4,133,43,574).  visible(lb2gt_4,574). happensAt(walking(lb2gt_4),574). orientation(lb2gt_6,151,574). coords(lb2gt_6,72,122,574).  visible(lb2gt_6,574). happensAt(inactive(lb2gt_6),574).
orientation(lb2gt_4,154,575). coords(lb2gt_4,133,42,575).  visible(lb2gt_4,575). happensAt(walking(lb2gt_4),575). orientation(lb2gt_6,151,575). coords(lb2gt_6,72,122,575).  visible(lb2gt_6,575). happensAt(inactive(lb2gt_6),575).
orientation(lb2gt_4,154,576). coords(lb2gt_4,134,41,576).  visible(lb2gt_4,576). happensAt(walking(lb2gt_4),576). orientation(lb2gt_6,151,576). coords(lb2gt_6,72,122,576).  visible(lb2gt_6,576). happensAt(inactive(lb2gt_6),576).
orientation(lb2gt_4,154,577). coords(lb2gt_4,134,40,577).  visible(lb2gt_4,577). happensAt(walking(lb2gt_4),577). orientation(lb2gt_6,151,577). coords(lb2gt_6,72,122,577).  visible(lb2gt_6,577). happensAt(inactive(lb2gt_6),577).
orientation(lb2gt_4,154,578). coords(lb2gt_4,134,40,578).  visible(lb2gt_4,578). happensAt(walking(lb2gt_4),578). orientation(lb2gt_6,151,578). coords(lb2gt_6,72,122,578).  visible(lb2gt_6,578). happensAt(inactive(lb2gt_6),578).
orientation(lb2gt_4,154,579). coords(lb2gt_4,134,39,579).  visible(lb2gt_4,579). happensAt(walking(lb2gt_4),579). orientation(lb2gt_6,151,579). coords(lb2gt_6,72,122,579).  visible(lb2gt_6,579). happensAt(inactive(lb2gt_6),579).
orientation(lb2gt_4,154,580). coords(lb2gt_4,136,38,580).  visible(lb2gt_4,580). happensAt(walking(lb2gt_4),580). orientation(lb2gt_6,151,580). coords(lb2gt_6,72,122,580).  visible(lb2gt_6,580). happensAt(inactive(lb2gt_6),580).
orientation(lb2gt_4,154,581). coords(lb2gt_4,137,38,581).  visible(lb2gt_4,581). happensAt(walking(lb2gt_4),581). orientation(lb2gt_6,151,581). coords(lb2gt_6,72,122,581).  visible(lb2gt_6,581). happensAt(inactive(lb2gt_6),581).
orientation(lb2gt_4,154,582). coords(lb2gt_4,138,38,582).  visible(lb2gt_4,582). happensAt(walking(lb2gt_4),582). orientation(lb2gt_6,151,582). coords(lb2gt_6,72,122,582).  visible(lb2gt_6,582). happensAt(inactive(lb2gt_6),582).
orientation(lb2gt_4,154,583). coords(lb2gt_4,139,38,583).  visible(lb2gt_4,583). happensAt(walking(lb2gt_4),583). orientation(lb2gt_6,151,583). coords(lb2gt_6,72,122,583).  visible(lb2gt_6,583). happensAt(inactive(lb2gt_6),583).
orientation(lb2gt_4,154,584). coords(lb2gt_4,140,37,584).  visible(lb2gt_4,584). happensAt(walking(lb2gt_4),584). orientation(lb2gt_6,151,584). coords(lb2gt_6,72,122,584).  visible(lb2gt_6,584). happensAt(inactive(lb2gt_6),584).
orientation(lb2gt_4,154,585). coords(lb2gt_4,140,37,585).  visible(lb2gt_4,585). happensAt(walking(lb2gt_4),585). orientation(lb2gt_6,151,585). coords(lb2gt_6,72,122,585).  visible(lb2gt_6,585). happensAt(inactive(lb2gt_6),585).
orientation(lb2gt_4,130,586). coords(lb2gt_4,141,37,586).  visible(lb2gt_4,586). happensAt(walking(lb2gt_4),586). orientation(lb2gt_6,151,586). coords(lb2gt_6,72,122,586).  visible(lb2gt_6,586). happensAt(inactive(lb2gt_6),586).
orientation(lb2gt_4,130,587). coords(lb2gt_4,142,36,587).  visible(lb2gt_4,587). happensAt(walking(lb2gt_4),587). orientation(lb2gt_6,151,587). coords(lb2gt_6,72,122,587).  visible(lb2gt_6,587). happensAt(inactive(lb2gt_6),587).
#show holdsAt/2.