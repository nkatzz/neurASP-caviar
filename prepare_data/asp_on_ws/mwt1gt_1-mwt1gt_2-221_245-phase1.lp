
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

	orientation(mwt1gt_1,160,221). coords(mwt1gt_1,55,187,221).  visible(mwt1gt_1,221). happensAt(walking(mwt1gt_1),221). orientation(mwt1gt_2,140,221). coords(mwt1gt_2,283,7,221).  happensAt(appear(mwt1gt_2),221). happensAt(walking(mwt1gt_2),221).
orientation(mwt1gt_1,160,222). coords(mwt1gt_1,56,186,222).  visible(mwt1gt_1,222). happensAt(walking(mwt1gt_1),222). orientation(mwt1gt_2,140,222). coords(mwt1gt_2,283,9,222).  visible(mwt1gt_2,222). happensAt(walking(mwt1gt_2),222).
orientation(mwt1gt_1,160,223). coords(mwt1gt_1,56,186,223).  visible(mwt1gt_1,223). happensAt(walking(mwt1gt_1),223). orientation(mwt1gt_2,110,223). coords(mwt1gt_2,283,12,223).  visible(mwt1gt_2,223). happensAt(walking(mwt1gt_2),223).
orientation(mwt1gt_1,160,224). coords(mwt1gt_1,58,186,224).  visible(mwt1gt_1,224). happensAt(walking(mwt1gt_1),224). orientation(mwt1gt_2,110,224). coords(mwt1gt_2,282,12,224).  visible(mwt1gt_2,224). happensAt(walking(mwt1gt_2),224).
orientation(mwt1gt_1,160,225). coords(mwt1gt_1,58,186,225).  visible(mwt1gt_1,225). happensAt(walking(mwt1gt_1),225). orientation(mwt1gt_2,110,225). coords(mwt1gt_2,280,12,225).  visible(mwt1gt_2,225). happensAt(walking(mwt1gt_2),225).
orientation(mwt1gt_1,160,226). coords(mwt1gt_1,58,185,226).  visible(mwt1gt_1,226). happensAt(walking(mwt1gt_1),226). orientation(mwt1gt_2,110,226). coords(mwt1gt_2,280,13,226).  visible(mwt1gt_2,226). happensAt(walking(mwt1gt_2),226).
orientation(mwt1gt_1,160,227). coords(mwt1gt_1,58,185,227).  visible(mwt1gt_1,227). happensAt(walking(mwt1gt_1),227). orientation(mwt1gt_2,110,227). coords(mwt1gt_2,280,13,227).  visible(mwt1gt_2,227). happensAt(walking(mwt1gt_2),227).
orientation(mwt1gt_1,160,228). coords(mwt1gt_1,58,185,228).  visible(mwt1gt_1,228). happensAt(walking(mwt1gt_1),228). orientation(mwt1gt_2,110,228). coords(mwt1gt_2,280,13,228).  visible(mwt1gt_2,228). happensAt(walking(mwt1gt_2),228).
orientation(mwt1gt_1,160,229). coords(mwt1gt_1,58,183,229).  visible(mwt1gt_1,229). happensAt(walking(mwt1gt_1),229). orientation(mwt1gt_2,110,229). coords(mwt1gt_2,280,13,229).  visible(mwt1gt_2,229). happensAt(walking(mwt1gt_2),229).
orientation(mwt1gt_1,160,230). coords(mwt1gt_1,58,183,230).  visible(mwt1gt_1,230). happensAt(walking(mwt1gt_1),230). orientation(mwt1gt_2,110,230). coords(mwt1gt_2,280,13,230).  visible(mwt1gt_2,230). happensAt(walking(mwt1gt_2),230).
orientation(mwt1gt_1,160,231). coords(mwt1gt_1,58,183,231).  visible(mwt1gt_1,231). happensAt(walking(mwt1gt_1),231). orientation(mwt1gt_2,110,231). coords(mwt1gt_2,280,13,231).  visible(mwt1gt_2,231). happensAt(walking(mwt1gt_2),231).
orientation(mwt1gt_1,160,232). coords(mwt1gt_1,58,183,232).  visible(mwt1gt_1,232). happensAt(walking(mwt1gt_1),232). orientation(mwt1gt_2,110,232). coords(mwt1gt_2,275,13,232).  visible(mwt1gt_2,232). happensAt(walking(mwt1gt_2),232).
orientation(mwt1gt_1,160,233). coords(mwt1gt_1,58,183,233).  visible(mwt1gt_1,233). happensAt(walking(mwt1gt_1),233). orientation(mwt1gt_2,110,233). coords(mwt1gt_2,272,14,233).  visible(mwt1gt_2,233). happensAt(walking(mwt1gt_2),233).
orientation(mwt1gt_1,160,234). coords(mwt1gt_1,60,184,234).  visible(mwt1gt_1,234). happensAt(walking(mwt1gt_1),234). orientation(mwt1gt_2,110,234). coords(mwt1gt_2,272,14,234).  visible(mwt1gt_2,234). happensAt(walking(mwt1gt_2),234).
orientation(mwt1gt_1,160,235). coords(mwt1gt_1,62,184,235).  visible(mwt1gt_1,235). happensAt(walking(mwt1gt_1),235). orientation(mwt1gt_2,110,235). coords(mwt1gt_2,272,16,235).  visible(mwt1gt_2,235). happensAt(walking(mwt1gt_2),235).
orientation(mwt1gt_1,160,236). coords(mwt1gt_1,62,184,236).  visible(mwt1gt_1,236). happensAt(walking(mwt1gt_1),236). orientation(mwt1gt_2,110,236). coords(mwt1gt_2,272,17,236).  visible(mwt1gt_2,236). happensAt(walking(mwt1gt_2),236).
orientation(mwt1gt_1,160,237). coords(mwt1gt_1,62,184,237).  visible(mwt1gt_1,237). happensAt(walking(mwt1gt_1),237). orientation(mwt1gt_2,110,237). coords(mwt1gt_2,270,19,237).  visible(mwt1gt_2,237). happensAt(walking(mwt1gt_2),237).
orientation(mwt1gt_1,160,238). coords(mwt1gt_1,62,184,238).  visible(mwt1gt_1,238). happensAt(walking(mwt1gt_1),238). orientation(mwt1gt_2,110,238). coords(mwt1gt_2,270,19,238).  visible(mwt1gt_2,238). happensAt(walking(mwt1gt_2),238).
orientation(mwt1gt_1,160,239). coords(mwt1gt_1,62,184,239).  visible(mwt1gt_1,239). happensAt(walking(mwt1gt_1),239). orientation(mwt1gt_2,110,239). coords(mwt1gt_2,266,20,239).  visible(mwt1gt_2,239). happensAt(walking(mwt1gt_2),239).
orientation(mwt1gt_1,160,240). coords(mwt1gt_1,62,184,240).  visible(mwt1gt_1,240). happensAt(walking(mwt1gt_1),240). orientation(mwt1gt_2,110,240). coords(mwt1gt_2,261,21,240).  visible(mwt1gt_2,240). happensAt(walking(mwt1gt_2),240).
orientation(mwt1gt_1,160,241). coords(mwt1gt_1,62,184,241).  visible(mwt1gt_1,241). happensAt(walking(mwt1gt_1),241). orientation(mwt1gt_2,110,241). coords(mwt1gt_2,261,21,241).  visible(mwt1gt_2,241). happensAt(walking(mwt1gt_2),241).
orientation(mwt1gt_1,160,242). coords(mwt1gt_1,62,184,242).  visible(mwt1gt_1,242). happensAt(walking(mwt1gt_1),242). orientation(mwt1gt_2,110,242). coords(mwt1gt_2,258,22,242).  visible(mwt1gt_2,242). happensAt(walking(mwt1gt_2),242).
orientation(mwt1gt_1,160,243). coords(mwt1gt_1,64,184,243).  visible(mwt1gt_1,243). happensAt(walking(mwt1gt_1),243). orientation(mwt1gt_2,110,243). coords(mwt1gt_2,256,22,243).  visible(mwt1gt_2,243). happensAt(walking(mwt1gt_2),243).
orientation(mwt1gt_1,160,244). coords(mwt1gt_1,65,184,244).  visible(mwt1gt_1,244). happensAt(walking(mwt1gt_1),244). orientation(mwt1gt_2,110,244). coords(mwt1gt_2,256,22,244).  visible(mwt1gt_2,244). happensAt(walking(mwt1gt_2),244).
orientation(mwt1gt_1,160,245). coords(mwt1gt_1,65,184,245).  visible(mwt1gt_1,245). happensAt(walking(mwt1gt_1),245). orientation(mwt1gt_2,110,245). coords(mwt1gt_2,251,22,245).  visible(mwt1gt_2,245). happensAt(walking(mwt1gt_2),245).
#show holdsAt/2.