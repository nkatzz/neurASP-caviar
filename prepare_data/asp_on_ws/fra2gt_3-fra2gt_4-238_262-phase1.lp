
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

	orientation(fra2gt_3,151,238). coords(fra2gt_3,55,153,238).  visible(fra2gt_3,238). happensAt(walking(fra2gt_3),238). orientation(fra2gt_4,159,238). coords(fra2gt_4,179,183,238).  visible(fra2gt_4,238). happensAt(walking(fra2gt_4),238).
orientation(fra2gt_3,151,239). coords(fra2gt_3,55,152,239).  visible(fra2gt_3,239). happensAt(walking(fra2gt_3),239). orientation(fra2gt_4,159,239). coords(fra2gt_4,178,182,239).  visible(fra2gt_4,239). happensAt(walking(fra2gt_4),239).
orientation(fra2gt_3,151,240). coords(fra2gt_3,56,151,240).  visible(fra2gt_3,240). happensAt(walking(fra2gt_3),240). orientation(fra2gt_4,159,240). coords(fra2gt_4,177,181,240).  visible(fra2gt_4,240). happensAt(walking(fra2gt_4),240).
orientation(fra2gt_3,151,241). coords(fra2gt_3,56,149,241).  visible(fra2gt_3,241). happensAt(walking(fra2gt_3),241). orientation(fra2gt_4,159,241). coords(fra2gt_4,177,180,241).  visible(fra2gt_4,241). happensAt(walking(fra2gt_4),241).
orientation(fra2gt_3,151,242). coords(fra2gt_3,57,149,242).  visible(fra2gt_3,242). happensAt(walking(fra2gt_3),242). orientation(fra2gt_4,159,242). coords(fra2gt_4,176,179,242).  visible(fra2gt_4,242). happensAt(walking(fra2gt_4),242).
orientation(fra2gt_3,151,243). coords(fra2gt_3,58,148,243).  visible(fra2gt_3,243). happensAt(walking(fra2gt_3),243). orientation(fra2gt_4,155,243). coords(fra2gt_4,175,178,243).  visible(fra2gt_4,243). happensAt(walking(fra2gt_4),243).
orientation(fra2gt_3,151,244). coords(fra2gt_3,60,148,244).  visible(fra2gt_3,244). happensAt(walking(fra2gt_3),244). orientation(fra2gt_4,155,244). coords(fra2gt_4,174,176,244).  visible(fra2gt_4,244). happensAt(walking(fra2gt_4),244).
orientation(fra2gt_3,151,245). coords(fra2gt_3,60,148,245).  visible(fra2gt_3,245). happensAt(walking(fra2gt_3),245). orientation(fra2gt_4,155,245). coords(fra2gt_4,173,174,245).  visible(fra2gt_4,245). happensAt(walking(fra2gt_4),245).
orientation(fra2gt_3,151,246). coords(fra2gt_3,60,148,246).  visible(fra2gt_3,246). happensAt(walking(fra2gt_3),246). orientation(fra2gt_4,155,246). coords(fra2gt_4,171,172,246).  visible(fra2gt_4,246). happensAt(walking(fra2gt_4),246).
orientation(fra2gt_3,151,247). coords(fra2gt_3,60,147,247).  visible(fra2gt_3,247). happensAt(walking(fra2gt_3),247). orientation(fra2gt_4,155,247). coords(fra2gt_4,168,169,247).  visible(fra2gt_4,247). happensAt(walking(fra2gt_4),247).
orientation(fra2gt_3,151,248). coords(fra2gt_3,61,146,248).  visible(fra2gt_3,248). happensAt(walking(fra2gt_3),248). orientation(fra2gt_4,155,248). coords(fra2gt_4,166,167,248).  visible(fra2gt_4,248). happensAt(walking(fra2gt_4),248).
orientation(fra2gt_3,151,249). coords(fra2gt_3,61,145,249).  visible(fra2gt_3,249). happensAt(walking(fra2gt_3),249). orientation(fra2gt_4,155,249). coords(fra2gt_4,164,163,249).  visible(fra2gt_4,249). happensAt(walking(fra2gt_4),249).
orientation(fra2gt_3,151,250). coords(fra2gt_3,62,144,250).  visible(fra2gt_3,250). happensAt(walking(fra2gt_3),250). orientation(fra2gt_4,155,250). coords(fra2gt_4,163,160,250).  visible(fra2gt_4,250). happensAt(walking(fra2gt_4),250).
orientation(fra2gt_3,151,251). coords(fra2gt_3,61,142,251).  visible(fra2gt_3,251). happensAt(walking(fra2gt_3),251). orientation(fra2gt_4,155,251). coords(fra2gt_4,162,158,251).  visible(fra2gt_4,251). happensAt(walking(fra2gt_4),251).
orientation(fra2gt_3,151,252). coords(fra2gt_3,61,142,252).  visible(fra2gt_3,252). happensAt(walking(fra2gt_3),252). orientation(fra2gt_4,155,252). coords(fra2gt_4,162,157,252).  visible(fra2gt_4,252). happensAt(walking(fra2gt_4),252).
orientation(fra2gt_3,151,253). coords(fra2gt_3,63,142,253).  visible(fra2gt_3,253). happensAt(walking(fra2gt_3),253). orientation(fra2gt_4,155,253). coords(fra2gt_4,161,155,253).  visible(fra2gt_4,253). happensAt(walking(fra2gt_4),253).
orientation(fra2gt_3,151,254). coords(fra2gt_3,64,141,254).  visible(fra2gt_3,254). happensAt(walking(fra2gt_3),254). orientation(fra2gt_4,155,254). coords(fra2gt_4,160,154,254).  visible(fra2gt_4,254). happensAt(walking(fra2gt_4),254).
orientation(fra2gt_3,151,255). coords(fra2gt_3,65,141,255).  visible(fra2gt_3,255). happensAt(walking(fra2gt_3),255). orientation(fra2gt_4,155,255). coords(fra2gt_4,160,153,255).  visible(fra2gt_4,255). happensAt(walking(fra2gt_4),255).
orientation(fra2gt_3,151,256). coords(fra2gt_3,66,141,256).  visible(fra2gt_3,256). happensAt(walking(fra2gt_3),256). orientation(fra2gt_4,155,256). coords(fra2gt_4,159,152,256).  visible(fra2gt_4,256). happensAt(walking(fra2gt_4),256).
orientation(fra2gt_3,151,257). coords(fra2gt_3,66,141,257).  visible(fra2gt_3,257). happensAt(walking(fra2gt_3),257). orientation(fra2gt_4,155,257). coords(fra2gt_4,158,151,257).  visible(fra2gt_4,257). happensAt(walking(fra2gt_4),257).
orientation(fra2gt_3,151,258). coords(fra2gt_3,67,140,258).  visible(fra2gt_3,258). happensAt(walking(fra2gt_3),258). orientation(fra2gt_4,155,258). coords(fra2gt_4,156,149,258).  visible(fra2gt_4,258). happensAt(walking(fra2gt_4),258).
orientation(fra2gt_3,151,259). coords(fra2gt_3,67,139,259).  visible(fra2gt_3,259). happensAt(walking(fra2gt_3),259). orientation(fra2gt_4,155,259). coords(fra2gt_4,153,148,259).  visible(fra2gt_4,259). happensAt(walking(fra2gt_4),259).
orientation(fra2gt_3,151,260). coords(fra2gt_3,67,138,260).  visible(fra2gt_3,260). happensAt(walking(fra2gt_3),260). orientation(fra2gt_4,155,260). coords(fra2gt_4,151,147,260).  visible(fra2gt_4,260). happensAt(walking(fra2gt_4),260).
orientation(fra2gt_3,151,261). coords(fra2gt_3,67,137,261).  visible(fra2gt_3,261). happensAt(walking(fra2gt_3),261). orientation(fra2gt_4,155,261). coords(fra2gt_4,149,147,261).  visible(fra2gt_4,261). happensAt(walking(fra2gt_4),261).
orientation(fra2gt_3,151,262). coords(fra2gt_3,67,136,262).  visible(fra2gt_3,262). happensAt(walking(fra2gt_3),262). orientation(fra2gt_4,155,262). coords(fra2gt_4,148,146,262).  visible(fra2gt_4,262). happensAt(walking(fra2gt_4),262).
#show holdsAt/2.