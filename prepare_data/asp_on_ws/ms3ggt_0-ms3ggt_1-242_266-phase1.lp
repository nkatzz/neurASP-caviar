
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

	orientation(ms3ggt_0,154,242). coords(ms3ggt_0,58,155,242).  visible(ms3ggt_0,242). happensAt(inactive(ms3ggt_0),242). orientation(ms3ggt_1,167,242). coords(ms3ggt_1,52,165,242).  visible(ms3ggt_1,242). happensAt(walking(ms3ggt_1),242).
orientation(ms3ggt_0,154,243). coords(ms3ggt_0,58,155,243).  visible(ms3ggt_0,243). happensAt(inactive(ms3ggt_0),243). orientation(ms3ggt_1,167,243). coords(ms3ggt_1,52,165,243).  visible(ms3ggt_1,243). happensAt(walking(ms3ggt_1),243).
orientation(ms3ggt_0,154,244). coords(ms3ggt_0,58,155,244).  visible(ms3ggt_0,244). happensAt(inactive(ms3ggt_0),244). orientation(ms3ggt_1,167,244). coords(ms3ggt_1,52,165,244).  visible(ms3ggt_1,244). happensAt(walking(ms3ggt_1),244).
orientation(ms3ggt_0,154,245). coords(ms3ggt_0,58,154,245).  visible(ms3ggt_0,245). happensAt(inactive(ms3ggt_0),245). orientation(ms3ggt_1,167,245). coords(ms3ggt_1,52,165,245).  visible(ms3ggt_1,245). happensAt(walking(ms3ggt_1),245).
orientation(ms3ggt_0,154,246). coords(ms3ggt_0,58,154,246).  visible(ms3ggt_0,246). happensAt(inactive(ms3ggt_0),246). orientation(ms3ggt_1,167,246). coords(ms3ggt_1,52,165,246).  visible(ms3ggt_1,246). happensAt(walking(ms3ggt_1),246).
orientation(ms3ggt_0,154,247). coords(ms3ggt_0,58,154,247).  visible(ms3ggt_0,247). happensAt(inactive(ms3ggt_0),247). orientation(ms3ggt_1,167,247). coords(ms3ggt_1,52,165,247).  visible(ms3ggt_1,247). happensAt(walking(ms3ggt_1),247).
orientation(ms3ggt_0,154,248). coords(ms3ggt_0,58,154,248).  visible(ms3ggt_0,248). happensAt(inactive(ms3ggt_0),248). orientation(ms3ggt_1,167,248). coords(ms3ggt_1,52,165,248).  visible(ms3ggt_1,248). happensAt(walking(ms3ggt_1),248).
orientation(ms3ggt_0,154,249). coords(ms3ggt_0,58,154,249).  visible(ms3ggt_0,249). happensAt(inactive(ms3ggt_0),249). orientation(ms3ggt_1,167,249). coords(ms3ggt_1,53,165,249).  visible(ms3ggt_1,249). happensAt(walking(ms3ggt_1),249).
orientation(ms3ggt_0,154,250). coords(ms3ggt_0,58,154,250).  visible(ms3ggt_0,250). happensAt(inactive(ms3ggt_0),250). orientation(ms3ggt_1,167,250). coords(ms3ggt_1,53,165,250).  visible(ms3ggt_1,250). happensAt(walking(ms3ggt_1),250).
orientation(ms3ggt_0,154,251). coords(ms3ggt_0,58,154,251).  visible(ms3ggt_0,251). happensAt(inactive(ms3ggt_0),251). orientation(ms3ggt_1,167,251). coords(ms3ggt_1,53,165,251).  visible(ms3ggt_1,251). happensAt(walking(ms3ggt_1),251).
orientation(ms3ggt_0,154,252). coords(ms3ggt_0,58,154,252).  visible(ms3ggt_0,252). happensAt(inactive(ms3ggt_0),252). orientation(ms3ggt_1,167,252). coords(ms3ggt_1,53,165,252).  visible(ms3ggt_1,252). happensAt(walking(ms3ggt_1),252).
orientation(ms3ggt_0,154,253). coords(ms3ggt_0,58,153,253).  visible(ms3ggt_0,253). happensAt(walking(ms3ggt_0),253). orientation(ms3ggt_1,167,253). coords(ms3ggt_1,54,165,253).  visible(ms3ggt_1,253). happensAt(walking(ms3ggt_1),253).
orientation(ms3ggt_0,154,254). coords(ms3ggt_0,58,153,254).  visible(ms3ggt_0,254). happensAt(walking(ms3ggt_0),254). orientation(ms3ggt_1,167,254). coords(ms3ggt_1,55,165,254).  visible(ms3ggt_1,254). happensAt(walking(ms3ggt_1),254).
orientation(ms3ggt_0,154,255). coords(ms3ggt_0,59,153,255).  visible(ms3ggt_0,255). happensAt(walking(ms3ggt_0),255). orientation(ms3ggt_1,167,255). coords(ms3ggt_1,55,166,255).  visible(ms3ggt_1,255). happensAt(walking(ms3ggt_1),255).
orientation(ms3ggt_0,154,256). coords(ms3ggt_0,59,153,256).  visible(ms3ggt_0,256). happensAt(walking(ms3ggt_0),256). orientation(ms3ggt_1,167,256). coords(ms3ggt_1,55,166,256).  visible(ms3ggt_1,256). happensAt(walking(ms3ggt_1),256).
orientation(ms3ggt_0,154,257). coords(ms3ggt_0,59,153,257).  visible(ms3ggt_0,257). happensAt(walking(ms3ggt_0),257). orientation(ms3ggt_1,155,257). coords(ms3ggt_1,55,166,257).  visible(ms3ggt_1,257). happensAt(walking(ms3ggt_1),257).
orientation(ms3ggt_0,154,258). coords(ms3ggt_0,60,153,258).  visible(ms3ggt_0,258). happensAt(walking(ms3ggt_0),258). orientation(ms3ggt_1,155,258). coords(ms3ggt_1,55,166,258).  visible(ms3ggt_1,258). happensAt(walking(ms3ggt_1),258).
orientation(ms3ggt_0,154,259). coords(ms3ggt_0,60,153,259).  visible(ms3ggt_0,259). happensAt(walking(ms3ggt_0),259). orientation(ms3ggt_1,155,259). coords(ms3ggt_1,56,166,259).  visible(ms3ggt_1,259). happensAt(walking(ms3ggt_1),259).
orientation(ms3ggt_0,154,260). coords(ms3ggt_0,62,152,260).  visible(ms3ggt_0,260). happensAt(walking(ms3ggt_0),260). orientation(ms3ggt_1,155,260). coords(ms3ggt_1,56,166,260).  visible(ms3ggt_1,260). happensAt(walking(ms3ggt_1),260).
orientation(ms3ggt_0,154,261). coords(ms3ggt_0,62,152,261).  visible(ms3ggt_0,261). happensAt(walking(ms3ggt_0),261). orientation(ms3ggt_1,155,261). coords(ms3ggt_1,57,166,261).  visible(ms3ggt_1,261). happensAt(walking(ms3ggt_1),261).
orientation(ms3ggt_0,154,262). coords(ms3ggt_0,62,152,262).  visible(ms3ggt_0,262). happensAt(walking(ms3ggt_0),262). orientation(ms3ggt_1,155,262). coords(ms3ggt_1,59,166,262).  visible(ms3ggt_1,262). happensAt(walking(ms3ggt_1),262).
orientation(ms3ggt_0,157,263). coords(ms3ggt_0,63,151,263).  visible(ms3ggt_0,263). happensAt(walking(ms3ggt_0),263). orientation(ms3ggt_1,158,263). coords(ms3ggt_1,60,166,263).  visible(ms3ggt_1,263). happensAt(walking(ms3ggt_1),263).
orientation(ms3ggt_0,157,264). coords(ms3ggt_0,63,151,264).  visible(ms3ggt_0,264). happensAt(walking(ms3ggt_0),264). orientation(ms3ggt_1,158,264). coords(ms3ggt_1,61,166,264).  visible(ms3ggt_1,264). happensAt(walking(ms3ggt_1),264).
orientation(ms3ggt_0,157,265). coords(ms3ggt_0,63,151,265).  visible(ms3ggt_0,265). happensAt(walking(ms3ggt_0),265). orientation(ms3ggt_1,158,265). coords(ms3ggt_1,61,165,265).  visible(ms3ggt_1,265). happensAt(walking(ms3ggt_1),265).
orientation(ms3ggt_0,157,266). coords(ms3ggt_0,63,151,266).  visible(ms3ggt_0,266). happensAt(walking(ms3ggt_0),266). orientation(ms3ggt_1,158,266). coords(ms3ggt_1,62,165,266).  visible(ms3ggt_1,266). happensAt(walking(ms3ggt_1),266).
#show holdsAt/2.