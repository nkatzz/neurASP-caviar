
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

	orientation(fomdgt3_2,130,567). coords(fomdgt3_2,249,91,567).  visible(fomdgt3_2,567). happensAt(walking(fomdgt3_2),567). orientation(fomdgt3_3,170,567). coords(fomdgt3_3,46,103,567).  happensAt(appear(fomdgt3_3),567). happensAt(walking(fomdgt3_3),567).
orientation(fomdgt3_2,130,568). coords(fomdgt3_2,249,91,568).  visible(fomdgt3_2,568). happensAt(walking(fomdgt3_2),568). orientation(fomdgt3_3,170,568). coords(fomdgt3_3,46,103,568).  visible(fomdgt3_3,568). happensAt(walking(fomdgt3_3),568).
orientation(fomdgt3_2,130,569). coords(fomdgt3_2,248,91,569).  visible(fomdgt3_2,569). happensAt(walking(fomdgt3_2),569). orientation(fomdgt3_3,170,569). coords(fomdgt3_3,46,103,569).  visible(fomdgt3_3,569). happensAt(walking(fomdgt3_3),569).
orientation(fomdgt3_2,130,570). coords(fomdgt3_2,248,92,570).  visible(fomdgt3_2,570). happensAt(walking(fomdgt3_2),570). orientation(fomdgt3_3,170,570). coords(fomdgt3_3,46,103,570).  visible(fomdgt3_3,570). happensAt(walking(fomdgt3_3),570).
orientation(fomdgt3_2,130,571). coords(fomdgt3_2,248,92,571).  visible(fomdgt3_2,571). happensAt(walking(fomdgt3_2),571). orientation(fomdgt3_3,170,571). coords(fomdgt3_3,46,103,571).  visible(fomdgt3_3,571). happensAt(walking(fomdgt3_3),571).
orientation(fomdgt3_2,130,572). coords(fomdgt3_2,248,92,572).  visible(fomdgt3_2,572). happensAt(walking(fomdgt3_2),572). orientation(fomdgt3_3,170,572). coords(fomdgt3_3,46,103,572).  visible(fomdgt3_3,572). happensAt(walking(fomdgt3_3),572).
orientation(fomdgt3_2,130,573). coords(fomdgt3_2,247,92,573).  visible(fomdgt3_2,573). happensAt(walking(fomdgt3_2),573). orientation(fomdgt3_3,170,573). coords(fomdgt3_3,46,103,573).  visible(fomdgt3_3,573). happensAt(walking(fomdgt3_3),573).
orientation(fomdgt3_2,130,574). coords(fomdgt3_2,247,92,574).  visible(fomdgt3_2,574). happensAt(walking(fomdgt3_2),574). orientation(fomdgt3_3,170,574). coords(fomdgt3_3,46,103,574).  visible(fomdgt3_3,574). happensAt(walking(fomdgt3_3),574).
orientation(fomdgt3_2,130,575). coords(fomdgt3_2,247,92,575).  visible(fomdgt3_2,575). happensAt(walking(fomdgt3_2),575). orientation(fomdgt3_3,159,575). coords(fomdgt3_3,50,101,575).  visible(fomdgt3_3,575). happensAt(walking(fomdgt3_3),575).
orientation(fomdgt3_2,130,576). coords(fomdgt3_2,247,93,576).  visible(fomdgt3_2,576). happensAt(walking(fomdgt3_2),576). orientation(fomdgt3_3,159,576). coords(fomdgt3_3,50,101,576).  visible(fomdgt3_3,576). happensAt(walking(fomdgt3_3),576).
orientation(fomdgt3_2,130,577). coords(fomdgt3_2,247,94,577).  visible(fomdgt3_2,577). happensAt(walking(fomdgt3_2),577). orientation(fomdgt3_3,159,577). coords(fomdgt3_3,50,101,577).  visible(fomdgt3_3,577). happensAt(walking(fomdgt3_3),577).
orientation(fomdgt3_2,130,578). coords(fomdgt3_2,248,94,578).  visible(fomdgt3_2,578). happensAt(walking(fomdgt3_2),578). orientation(fomdgt3_3,159,578). coords(fomdgt3_3,50,101,578).  visible(fomdgt3_3,578). happensAt(walking(fomdgt3_3),578).
orientation(fomdgt3_2,130,579). coords(fomdgt3_2,248,95,579).  visible(fomdgt3_2,579). happensAt(walking(fomdgt3_2),579). orientation(fomdgt3_3,159,579). coords(fomdgt3_3,50,101,579).  visible(fomdgt3_3,579). happensAt(walking(fomdgt3_3),579).
orientation(fomdgt3_2,130,580). coords(fomdgt3_2,249,97,580).  visible(fomdgt3_2,580). happensAt(walking(fomdgt3_2),580). orientation(fomdgt3_3,159,580). coords(fomdgt3_3,50,101,580).  visible(fomdgt3_3,580). happensAt(walking(fomdgt3_3),580).
orientation(fomdgt3_2,130,581). coords(fomdgt3_2,249,98,581).  visible(fomdgt3_2,581). happensAt(walking(fomdgt3_2),581). orientation(fomdgt3_3,159,581). coords(fomdgt3_3,50,101,581).  visible(fomdgt3_3,581). happensAt(walking(fomdgt3_3),581).
orientation(fomdgt3_2,130,582). coords(fomdgt3_2,249,99,582).  visible(fomdgt3_2,582). happensAt(walking(fomdgt3_2),582). orientation(fomdgt3_3,159,582). coords(fomdgt3_3,50,101,582).  visible(fomdgt3_3,582). happensAt(walking(fomdgt3_3),582).
orientation(fomdgt3_2,130,583). coords(fomdgt3_2,248,101,583).  visible(fomdgt3_2,583). happensAt(walking(fomdgt3_2),583). orientation(fomdgt3_3,159,583). coords(fomdgt3_3,52,100,583).  visible(fomdgt3_3,583). happensAt(walking(fomdgt3_3),583).
orientation(fomdgt3_2,130,584). coords(fomdgt3_2,249,101,584).  visible(fomdgt3_2,584). happensAt(walking(fomdgt3_2),584). orientation(fomdgt3_3,159,584). coords(fomdgt3_3,52,100,584).  visible(fomdgt3_3,584). happensAt(walking(fomdgt3_3),584).
orientation(fomdgt3_2,130,585). coords(fomdgt3_2,249,101,585).  visible(fomdgt3_2,585). happensAt(walking(fomdgt3_2),585). orientation(fomdgt3_3,159,585). coords(fomdgt3_3,52,100,585).  visible(fomdgt3_3,585). happensAt(walking(fomdgt3_3),585).
orientation(fomdgt3_2,130,586). coords(fomdgt3_2,249,102,586).  visible(fomdgt3_2,586). happensAt(walking(fomdgt3_2),586). orientation(fomdgt3_3,159,586). coords(fomdgt3_3,52,100,586).  visible(fomdgt3_3,586). happensAt(walking(fomdgt3_3),586).
orientation(fomdgt3_2,130,587). coords(fomdgt3_2,250,103,587).  visible(fomdgt3_2,587). happensAt(walking(fomdgt3_2),587). orientation(fomdgt3_3,159,587). coords(fomdgt3_3,52,100,587).  visible(fomdgt3_3,587). happensAt(walking(fomdgt3_3),587).
orientation(fomdgt3_2,130,588). coords(fomdgt3_2,250,103,588).  visible(fomdgt3_2,588). happensAt(walking(fomdgt3_2),588). orientation(fomdgt3_3,159,588). coords(fomdgt3_3,53,100,588).  visible(fomdgt3_3,588). happensAt(walking(fomdgt3_3),588).
orientation(fomdgt3_2,130,589). coords(fomdgt3_2,250,104,589).  visible(fomdgt3_2,589). happensAt(walking(fomdgt3_2),589). orientation(fomdgt3_3,159,589). coords(fomdgt3_3,55,99,589).  visible(fomdgt3_3,589). happensAt(walking(fomdgt3_3),589).
orientation(fomdgt3_2,130,590). coords(fomdgt3_2,250,105,590).  visible(fomdgt3_2,590). happensAt(walking(fomdgt3_2),590). orientation(fomdgt3_3,159,590). coords(fomdgt3_3,55,99,590).  visible(fomdgt3_3,590). happensAt(walking(fomdgt3_3),590).
orientation(fomdgt3_2,130,591). coords(fomdgt3_2,250,106,591).  visible(fomdgt3_2,591). happensAt(walking(fomdgt3_2),591). orientation(fomdgt3_3,159,591). coords(fomdgt3_3,57,99,591).  visible(fomdgt3_3,591). happensAt(walking(fomdgt3_3),591).
#show holdsAt/2.