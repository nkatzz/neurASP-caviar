
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

	orientation(rwgt_0,150,237). coords(rwgt_0,240,182,237).  visible(rwgt_0,237). happensAt(walking(rwgt_0),237). orientation(rwgt_1,132,237). coords(rwgt_1,164,29,237).  visible(rwgt_1,237). happensAt(walking(rwgt_1),237).
orientation(rwgt_0,150,238). coords(rwgt_0,243,184,238).  visible(rwgt_0,238). happensAt(walking(rwgt_0),238). orientation(rwgt_1,132,238). coords(rwgt_1,164,28,238).  visible(rwgt_1,238). happensAt(walking(rwgt_1),238).
orientation(rwgt_0,150,239). coords(rwgt_0,247,186,239).  visible(rwgt_0,239). happensAt(walking(rwgt_0),239). orientation(rwgt_1,132,239). coords(rwgt_1,165,27,239).  visible(rwgt_1,239). happensAt(walking(rwgt_1),239).
orientation(rwgt_0,150,240). coords(rwgt_0,251,188,240).  visible(rwgt_0,240). happensAt(walking(rwgt_0),240). orientation(rwgt_1,132,240). coords(rwgt_1,165,27,240).  visible(rwgt_1,240). happensAt(walking(rwgt_1),240).
orientation(rwgt_0,150,241). coords(rwgt_0,254,189,241).  visible(rwgt_0,241). happensAt(walking(rwgt_0),241). orientation(rwgt_1,132,241). coords(rwgt_1,165,27,241).  visible(rwgt_1,241). happensAt(walking(rwgt_1),241).
orientation(rwgt_0,150,242). coords(rwgt_0,257,190,242).  visible(rwgt_0,242). happensAt(walking(rwgt_0),242). orientation(rwgt_1,132,242). coords(rwgt_1,165,27,242).  visible(rwgt_1,242). happensAt(walking(rwgt_1),242).
orientation(rwgt_0,150,243). coords(rwgt_0,258,192,243).  visible(rwgt_0,243). happensAt(walking(rwgt_0),243). orientation(rwgt_1,132,243). coords(rwgt_1,166,26,243).  visible(rwgt_1,243). happensAt(walking(rwgt_1),243).
orientation(rwgt_0,150,244). coords(rwgt_0,260,193,244).  visible(rwgt_0,244). happensAt(walking(rwgt_0),244). orientation(rwgt_1,132,244). coords(rwgt_1,167,26,244).  visible(rwgt_1,244). happensAt(walking(rwgt_1),244).
orientation(rwgt_0,150,245). coords(rwgt_0,261,195,245).  visible(rwgt_0,245). happensAt(walking(rwgt_0),245). orientation(rwgt_1,132,245). coords(rwgt_1,167,26,245).  visible(rwgt_1,245). happensAt(walking(rwgt_1),245).
orientation(rwgt_0,150,246). coords(rwgt_0,262,197,246).  visible(rwgt_0,246). happensAt(walking(rwgt_0),246). orientation(rwgt_1,132,246). coords(rwgt_1,168,25,246).  visible(rwgt_1,246). happensAt(walking(rwgt_1),246).
orientation(rwgt_0,150,247). coords(rwgt_0,263,199,247).  visible(rwgt_0,247). happensAt(walking(rwgt_0),247). orientation(rwgt_1,132,247). coords(rwgt_1,168,24,247).  visible(rwgt_1,247). happensAt(walking(rwgt_1),247).
orientation(rwgt_0,149,248). coords(rwgt_0,264,202,248).  visible(rwgt_0,248). happensAt(walking(rwgt_0),248). orientation(rwgt_1,132,248). coords(rwgt_1,168,23,248).  visible(rwgt_1,248). happensAt(walking(rwgt_1),248).
orientation(rwgt_0,149,249). coords(rwgt_0,265,206,249).  visible(rwgt_0,249). happensAt(walking(rwgt_0),249). orientation(rwgt_1,132,249). coords(rwgt_1,168,23,249).  visible(rwgt_1,249). happensAt(walking(rwgt_1),249).
orientation(rwgt_0,153,250). coords(rwgt_0,267,209,250).  visible(rwgt_0,250). happensAt(walking(rwgt_0),250). orientation(rwgt_1,132,250). coords(rwgt_1,168,22,250).  visible(rwgt_1,250). happensAt(walking(rwgt_1),250).
orientation(rwgt_0,174,251). coords(rwgt_0,270,211,251).  visible(rwgt_0,251). happensAt(walking(rwgt_0),251). orientation(rwgt_1,132,251). coords(rwgt_1,168,22,251).  visible(rwgt_1,251). happensAt(walking(rwgt_1),251).
orientation(rwgt_0,167,252). coords(rwgt_0,273,216,252).  visible(rwgt_0,252). happensAt(walking(rwgt_0),252). orientation(rwgt_1,132,252). coords(rwgt_1,169,23,252).  visible(rwgt_1,252). happensAt(walking(rwgt_1),252).
orientation(rwgt_0,168,253). coords(rwgt_0,275,220,253).  visible(rwgt_0,253). happensAt(walking(rwgt_0),253). orientation(rwgt_1,132,253). coords(rwgt_1,169,22,253).  visible(rwgt_1,253). happensAt(walking(rwgt_1),253).
orientation(rwgt_0,8,254). coords(rwgt_0,278,223,254).  visible(rwgt_0,254). happensAt(walking(rwgt_0),254). orientation(rwgt_1,132,254). coords(rwgt_1,169,18,254).  visible(rwgt_1,254). happensAt(walking(rwgt_1),254).
orientation(rwgt_0,38,255). coords(rwgt_0,280,224,255).  visible(rwgt_0,255). happensAt(walking(rwgt_0),255). orientation(rwgt_1,132,255). coords(rwgt_1,169,16,255).  visible(rwgt_1,255). happensAt(walking(rwgt_1),255).
orientation(rwgt_0,45,256). coords(rwgt_0,281,225,256).  visible(rwgt_0,256). happensAt(walking(rwgt_0),256). orientation(rwgt_1,132,256). coords(rwgt_1,169,14,256).  visible(rwgt_1,256). happensAt(walking(rwgt_1),256).
orientation(rwgt_0,50,257). coords(rwgt_0,283,226,257).  visible(rwgt_0,257). happensAt(walking(rwgt_0),257). orientation(rwgt_1,132,257). coords(rwgt_1,169,13,257).  visible(rwgt_1,257). happensAt(walking(rwgt_1),257).
orientation(rwgt_0,51,258). coords(rwgt_0,283,226,258).  visible(rwgt_0,258). happensAt(walking(rwgt_0),258). orientation(rwgt_1,132,258). coords(rwgt_1,169,13,258).  visible(rwgt_1,258). happensAt(walking(rwgt_1),258).
orientation(rwgt_0,51,259). coords(rwgt_0,286,228,259).  visible(rwgt_0,259). happensAt(walking(rwgt_0),259). orientation(rwgt_1,132,259). coords(rwgt_1,169,13,259).  visible(rwgt_1,259). happensAt(walking(rwgt_1),259).
orientation(rwgt_0,51,260). coords(rwgt_0,288,232,260).  visible(rwgt_0,260). happensAt(walking(rwgt_0),260). orientation(rwgt_1,132,260). coords(rwgt_1,170,11,260).  visible(rwgt_1,260). happensAt(walking(rwgt_1),260).
orientation(rwgt_0,51,261). coords(rwgt_0,290,236,261).  visible(rwgt_0,261). happensAt(walking(rwgt_0),261). orientation(rwgt_1,132,261). coords(rwgt_1,170,11,261).  happensAt(disappear(rwgt_1),261). happensAt(walking(rwgt_1),261).
#show holdsAt/2.