
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

	orientation(fomdgt3_3,170,567). coords(fomdgt3_3,46,103,567).  happensAt(appear(fomdgt3_3),567). happensAt(walking(fomdgt3_3),567). orientation(fomdgt3_7,171,567). coords(fomdgt3_7,211,238,567).  visible(fomdgt3_7,567). happensAt(walking(fomdgt3_7),567).
orientation(fomdgt3_3,170,568). coords(fomdgt3_3,46,103,568).  visible(fomdgt3_3,568). happensAt(walking(fomdgt3_3),568). orientation(fomdgt3_7,171,568). coords(fomdgt3_7,211,238,568).  visible(fomdgt3_7,568). happensAt(walking(fomdgt3_7),568).
orientation(fomdgt3_3,170,569). coords(fomdgt3_3,46,103,569).  visible(fomdgt3_3,569). happensAt(walking(fomdgt3_3),569). orientation(fomdgt3_7,171,569). coords(fomdgt3_7,211,240,569).  visible(fomdgt3_7,569). happensAt(walking(fomdgt3_7),569).
orientation(fomdgt3_3,170,570). coords(fomdgt3_3,46,103,570).  visible(fomdgt3_3,570). happensAt(walking(fomdgt3_3),570). orientation(fomdgt3_7,171,570). coords(fomdgt3_7,212,243,570).  visible(fomdgt3_7,570). happensAt(walking(fomdgt3_7),570).
orientation(fomdgt3_3,170,571). coords(fomdgt3_3,46,103,571).  visible(fomdgt3_3,571). happensAt(walking(fomdgt3_3),571). orientation(fomdgt3_7,171,571). coords(fomdgt3_7,212,243,571).  visible(fomdgt3_7,571). happensAt(walking(fomdgt3_7),571).
orientation(fomdgt3_3,170,572). coords(fomdgt3_3,46,103,572).  visible(fomdgt3_3,572). happensAt(walking(fomdgt3_3),572). orientation(fomdgt3_7,171,572). coords(fomdgt3_7,213,249,572).  visible(fomdgt3_7,572). happensAt(walking(fomdgt3_7),572).
orientation(fomdgt3_3,170,573). coords(fomdgt3_3,46,103,573).  visible(fomdgt3_3,573). happensAt(walking(fomdgt3_3),573). orientation(fomdgt3_7,171,573). coords(fomdgt3_7,214,252,573).  visible(fomdgt3_7,573). happensAt(walking(fomdgt3_7),573).
orientation(fomdgt3_3,170,574). coords(fomdgt3_3,46,103,574).  visible(fomdgt3_3,574). happensAt(walking(fomdgt3_3),574). orientation(fomdgt3_7,171,574). coords(fomdgt3_7,216,256,574).  visible(fomdgt3_7,574). happensAt(walking(fomdgt3_7),574).
orientation(fomdgt3_3,159,575). coords(fomdgt3_3,50,101,575).  visible(fomdgt3_3,575). happensAt(walking(fomdgt3_3),575). orientation(fomdgt3_7,171,575). coords(fomdgt3_7,219,258,575).  visible(fomdgt3_7,575). happensAt(walking(fomdgt3_7),575).
orientation(fomdgt3_3,159,576). coords(fomdgt3_3,50,101,576).  visible(fomdgt3_3,576). happensAt(walking(fomdgt3_3),576). orientation(fomdgt3_7,171,576). coords(fomdgt3_7,222,260,576).  visible(fomdgt3_7,576). happensAt(walking(fomdgt3_7),576).
orientation(fomdgt3_3,159,577). coords(fomdgt3_3,50,101,577).  visible(fomdgt3_3,577). happensAt(walking(fomdgt3_3),577). orientation(fomdgt3_7,171,577). coords(fomdgt3_7,225,261,577).  visible(fomdgt3_7,577). happensAt(walking(fomdgt3_7),577).
orientation(fomdgt3_3,159,578). coords(fomdgt3_3,50,101,578).  visible(fomdgt3_3,578). happensAt(walking(fomdgt3_3),578). orientation(fomdgt3_7,171,578). coords(fomdgt3_7,225,262,578).  visible(fomdgt3_7,578). happensAt(walking(fomdgt3_7),578).
orientation(fomdgt3_3,159,579). coords(fomdgt3_3,50,101,579).  visible(fomdgt3_3,579). happensAt(walking(fomdgt3_3),579). orientation(fomdgt3_7,171,579). coords(fomdgt3_7,227,263,579).  visible(fomdgt3_7,579). happensAt(walking(fomdgt3_7),579).
orientation(fomdgt3_3,159,580). coords(fomdgt3_3,50,101,580).  visible(fomdgt3_3,580). happensAt(walking(fomdgt3_3),580). orientation(fomdgt3_7,171,580). coords(fomdgt3_7,227,263,580).  visible(fomdgt3_7,580). happensAt(walking(fomdgt3_7),580).
orientation(fomdgt3_3,159,581). coords(fomdgt3_3,50,101,581).  visible(fomdgt3_3,581). happensAt(walking(fomdgt3_3),581). orientation(fomdgt3_7,20,581). coords(fomdgt3_7,227,264,581).  visible(fomdgt3_7,581). happensAt(walking(fomdgt3_7),581).
orientation(fomdgt3_3,159,582). coords(fomdgt3_3,50,101,582).  visible(fomdgt3_3,582). happensAt(walking(fomdgt3_3),582). orientation(fomdgt3_7,20,582). coords(fomdgt3_7,227,264,582).  visible(fomdgt3_7,582). happensAt(walking(fomdgt3_7),582).
orientation(fomdgt3_3,159,583). coords(fomdgt3_3,52,100,583).  visible(fomdgt3_3,583). happensAt(walking(fomdgt3_3),583). orientation(fomdgt3_7,20,583). coords(fomdgt3_7,227,266,583).  visible(fomdgt3_7,583). happensAt(walking(fomdgt3_7),583).
orientation(fomdgt3_3,159,584). coords(fomdgt3_3,52,100,584).  visible(fomdgt3_3,584). happensAt(walking(fomdgt3_3),584). orientation(fomdgt3_7,20,584). coords(fomdgt3_7,229,268,584).  visible(fomdgt3_7,584). happensAt(walking(fomdgt3_7),584).
orientation(fomdgt3_3,159,585). coords(fomdgt3_3,52,100,585).  visible(fomdgt3_3,585). happensAt(walking(fomdgt3_3),585). orientation(fomdgt3_7,30,585). coords(fomdgt3_7,229,270,585).  visible(fomdgt3_7,585). happensAt(walking(fomdgt3_7),585).
orientation(fomdgt3_3,159,586). coords(fomdgt3_3,52,100,586).  visible(fomdgt3_3,586). happensAt(walking(fomdgt3_3),586). orientation(fomdgt3_7,30,586). coords(fomdgt3_7,229,272,586).  visible(fomdgt3_7,586). happensAt(walking(fomdgt3_7),586).
orientation(fomdgt3_3,159,587). coords(fomdgt3_3,52,100,587).  visible(fomdgt3_3,587). happensAt(walking(fomdgt3_3),587). orientation(fomdgt3_7,30,587). coords(fomdgt3_7,229,274,587).  visible(fomdgt3_7,587). happensAt(walking(fomdgt3_7),587).
orientation(fomdgt3_3,159,588). coords(fomdgt3_3,53,100,588).  visible(fomdgt3_3,588). happensAt(walking(fomdgt3_3),588). orientation(fomdgt3_7,30,588). coords(fomdgt3_7,231,274,588).  visible(fomdgt3_7,588). happensAt(walking(fomdgt3_7),588).
orientation(fomdgt3_3,159,589). coords(fomdgt3_3,55,99,589).  visible(fomdgt3_3,589). happensAt(walking(fomdgt3_3),589). orientation(fomdgt3_7,30,589). coords(fomdgt3_7,231,276,589).  visible(fomdgt3_7,589). happensAt(walking(fomdgt3_7),589).
orientation(fomdgt3_3,159,590). coords(fomdgt3_3,55,99,590).  visible(fomdgt3_3,590). happensAt(walking(fomdgt3_3),590). orientation(fomdgt3_7,30,590). coords(fomdgt3_7,231,276,590).  visible(fomdgt3_7,590). happensAt(walking(fomdgt3_7),590).
orientation(fomdgt3_3,159,591). coords(fomdgt3_3,57,99,591).  visible(fomdgt3_3,591). happensAt(walking(fomdgt3_3),591). orientation(fomdgt3_7,30,591). coords(fomdgt3_7,231,276,591).  visible(fomdgt3_7,591). happensAt(walking(fomdgt3_7),591).
#show holdsAt/2.