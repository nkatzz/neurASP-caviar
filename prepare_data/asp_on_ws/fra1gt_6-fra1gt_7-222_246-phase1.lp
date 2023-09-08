
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

	orientation(fra1gt_6,109,222). coords(fra1gt_6,260,19,222).  visible(fra1gt_6,222). happensAt(walking(fra1gt_6),222). orientation(fra1gt_7,127,222). coords(fra1gt_7,159,32,222).  visible(fra1gt_7,222). happensAt(walking(fra1gt_7),222).
orientation(fra1gt_6,109,223). coords(fra1gt_6,259,20,223).  visible(fra1gt_6,223). happensAt(walking(fra1gt_6),223). orientation(fra1gt_7,127,223). coords(fra1gt_7,159,32,223).  visible(fra1gt_7,223). happensAt(walking(fra1gt_7),223).
orientation(fra1gt_6,109,224). coords(fra1gt_6,258,21,224).  visible(fra1gt_6,224). happensAt(walking(fra1gt_6),224). orientation(fra1gt_7,127,224). coords(fra1gt_7,159,32,224).  visible(fra1gt_7,224). happensAt(walking(fra1gt_7),224).
orientation(fra1gt_6,109,225). coords(fra1gt_6,258,21,225).  visible(fra1gt_6,225). happensAt(walking(fra1gt_6),225). orientation(fra1gt_7,127,225). coords(fra1gt_7,159,33,225).  visible(fra1gt_7,225). happensAt(walking(fra1gt_7),225).
orientation(fra1gt_6,109,226). coords(fra1gt_6,258,22,226).  visible(fra1gt_6,226). happensAt(walking(fra1gt_6),226). orientation(fra1gt_7,127,226). coords(fra1gt_7,159,33,226).  visible(fra1gt_7,226). happensAt(walking(fra1gt_7),226).
orientation(fra1gt_6,109,227). coords(fra1gt_6,257,22,227).  visible(fra1gt_6,227). happensAt(walking(fra1gt_6),227). orientation(fra1gt_7,127,227). coords(fra1gt_7,158,33,227).  visible(fra1gt_7,227). happensAt(walking(fra1gt_7),227).
orientation(fra1gt_6,109,228). coords(fra1gt_6,256,22,228).  visible(fra1gt_6,228). happensAt(walking(fra1gt_6),228). orientation(fra1gt_7,127,228). coords(fra1gt_7,158,33,228).  visible(fra1gt_7,228). happensAt(walking(fra1gt_7),228).
orientation(fra1gt_6,109,229). coords(fra1gt_6,255,22,229).  visible(fra1gt_6,229). happensAt(walking(fra1gt_6),229). orientation(fra1gt_7,127,229). coords(fra1gt_7,157,34,229).  visible(fra1gt_7,229). happensAt(walking(fra1gt_7),229).
orientation(fra1gt_6,109,230). coords(fra1gt_6,253,22,230).  visible(fra1gt_6,230). happensAt(walking(fra1gt_6),230). orientation(fra1gt_7,127,230). coords(fra1gt_7,157,34,230).  visible(fra1gt_7,230). happensAt(walking(fra1gt_7),230).
orientation(fra1gt_6,109,231). coords(fra1gt_6,253,22,231).  visible(fra1gt_6,231). happensAt(walking(fra1gt_6),231). orientation(fra1gt_7,127,231). coords(fra1gt_7,157,35,231).  visible(fra1gt_7,231). happensAt(walking(fra1gt_7),231).
orientation(fra1gt_6,109,232). coords(fra1gt_6,252,22,232).  visible(fra1gt_6,232). happensAt(walking(fra1gt_6),232). orientation(fra1gt_7,127,232). coords(fra1gt_7,156,35,232).  visible(fra1gt_7,232). happensAt(walking(fra1gt_7),232).
orientation(fra1gt_6,109,233). coords(fra1gt_6,251,22,233).  visible(fra1gt_6,233). happensAt(walking(fra1gt_6),233). orientation(fra1gt_7,127,233). coords(fra1gt_7,156,35,233).  visible(fra1gt_7,233). happensAt(walking(fra1gt_7),233).
orientation(fra1gt_6,109,234). coords(fra1gt_6,251,22,234).  visible(fra1gt_6,234). happensAt(walking(fra1gt_6),234). orientation(fra1gt_7,127,234). coords(fra1gt_7,156,36,234).  visible(fra1gt_7,234). happensAt(walking(fra1gt_7),234).
orientation(fra1gt_6,109,235). coords(fra1gt_6,250,23,235).  visible(fra1gt_6,235). happensAt(walking(fra1gt_6),235). orientation(fra1gt_7,127,235). coords(fra1gt_7,156,36,235).  visible(fra1gt_7,235). happensAt(walking(fra1gt_7),235).
orientation(fra1gt_6,109,236). coords(fra1gt_6,250,26,236).  visible(fra1gt_6,236). happensAt(walking(fra1gt_6),236). orientation(fra1gt_7,132,236). coords(fra1gt_7,156,36,236).  visible(fra1gt_7,236). happensAt(walking(fra1gt_7),236).
orientation(fra1gt_6,109,237). coords(fra1gt_6,249,28,237).  visible(fra1gt_6,237). happensAt(walking(fra1gt_6),237). orientation(fra1gt_7,132,237). coords(fra1gt_7,156,36,237).  visible(fra1gt_7,237). happensAt(walking(fra1gt_7),237).
orientation(fra1gt_6,114,238). coords(fra1gt_6,248,30,238).  visible(fra1gt_6,238). happensAt(walking(fra1gt_6),238). orientation(fra1gt_7,137,238). coords(fra1gt_7,155,36,238).  visible(fra1gt_7,238). happensAt(walking(fra1gt_7),238).
orientation(fra1gt_6,114,239). coords(fra1gt_6,248,31,239).  visible(fra1gt_6,239). happensAt(walking(fra1gt_6),239). orientation(fra1gt_7,137,239). coords(fra1gt_7,155,36,239).  visible(fra1gt_7,239). happensAt(walking(fra1gt_7),239).
orientation(fra1gt_6,114,240). coords(fra1gt_6,247,33,240).  visible(fra1gt_6,240). happensAt(walking(fra1gt_6),240). orientation(fra1gt_7,137,240). coords(fra1gt_7,154,36,240).  visible(fra1gt_7,240). happensAt(walking(fra1gt_7),240).
orientation(fra1gt_6,114,241). coords(fra1gt_6,246,33,241).  visible(fra1gt_6,241). happensAt(walking(fra1gt_6),241). orientation(fra1gt_7,137,241). coords(fra1gt_7,154,36,241).  visible(fra1gt_7,241). happensAt(walking(fra1gt_7),241).
orientation(fra1gt_6,114,242). coords(fra1gt_6,245,34,242).  visible(fra1gt_6,242). happensAt(walking(fra1gt_6),242). orientation(fra1gt_7,137,242). coords(fra1gt_7,154,36,242).  visible(fra1gt_7,242). happensAt(walking(fra1gt_7),242).
orientation(fra1gt_6,119,243). coords(fra1gt_6,244,34,243).  visible(fra1gt_6,243). happensAt(walking(fra1gt_6),243). orientation(fra1gt_7,137,243). coords(fra1gt_7,153,37,243).  visible(fra1gt_7,243). happensAt(walking(fra1gt_7),243).
orientation(fra1gt_6,119,244). coords(fra1gt_6,243,35,244).  visible(fra1gt_6,244). happensAt(walking(fra1gt_6),244). orientation(fra1gt_7,137,244). coords(fra1gt_7,153,37,244).  visible(fra1gt_7,244). happensAt(walking(fra1gt_7),244).
orientation(fra1gt_6,123,245). coords(fra1gt_6,241,35,245).  visible(fra1gt_6,245). happensAt(walking(fra1gt_6),245). orientation(fra1gt_7,137,245). coords(fra1gt_7,151,37,245).  visible(fra1gt_7,245). happensAt(walking(fra1gt_7),245).
orientation(fra1gt_6,123,246). coords(fra1gt_6,240,35,246).  visible(fra1gt_6,246). happensAt(walking(fra1gt_6),246). orientation(fra1gt_7,137,246). coords(fra1gt_7,151,37,246).  visible(fra1gt_7,246). happensAt(walking(fra1gt_7),246).
#show holdsAt/2.