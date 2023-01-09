
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

	orientation(mc1gt_0,147,225). coords(mc1gt_0,153,130,225).  visible(mc1gt_0,225). happensAt(walking(mc1gt_0),225). orientation(mc1gt_2,148,225). coords(mc1gt_2,123,127,225).  visible(mc1gt_2,225). happensAt(walking(mc1gt_2),225).
orientation(mc1gt_0,147,226). coords(mc1gt_0,154,130,226).  visible(mc1gt_0,226). happensAt(walking(mc1gt_0),226). orientation(mc1gt_2,147,226). coords(mc1gt_2,123,127,226).  visible(mc1gt_2,226). happensAt(walking(mc1gt_2),226).
orientation(mc1gt_0,147,227). coords(mc1gt_0,155,131,227).  visible(mc1gt_0,227). happensAt(walking(mc1gt_0),227). orientation(mc1gt_2,147,227). coords(mc1gt_2,123,127,227).  visible(mc1gt_2,227). happensAt(walking(mc1gt_2),227).
orientation(mc1gt_0,147,228). coords(mc1gt_0,157,131,228).  visible(mc1gt_0,228). happensAt(walking(mc1gt_0),228). orientation(mc1gt_2,147,228). coords(mc1gt_2,123,129,228).  visible(mc1gt_2,228). happensAt(walking(mc1gt_2),228).
orientation(mc1gt_0,147,229). coords(mc1gt_0,159,132,229).  visible(mc1gt_0,229). happensAt(walking(mc1gt_0),229). orientation(mc1gt_2,147,229). coords(mc1gt_2,124,130,229).  visible(mc1gt_2,229). happensAt(walking(mc1gt_2),229).
orientation(mc1gt_0,147,230). coords(mc1gt_0,160,133,230).  visible(mc1gt_0,230). happensAt(walking(mc1gt_0),230). orientation(mc1gt_2,147,230). coords(mc1gt_2,125,131,230).  visible(mc1gt_2,230). happensAt(walking(mc1gt_2),230).
orientation(mc1gt_0,147,231). coords(mc1gt_0,161,133,231).  visible(mc1gt_0,231). happensAt(walking(mc1gt_0),231). orientation(mc1gt_2,146,231). coords(mc1gt_2,127,132,231).  visible(mc1gt_2,231). happensAt(walking(mc1gt_2),231).
orientation(mc1gt_0,147,232). coords(mc1gt_0,162,135,232).  visible(mc1gt_0,232). happensAt(walking(mc1gt_0),232). orientation(mc1gt_2,145,232). coords(mc1gt_2,129,133,232).  visible(mc1gt_2,232). happensAt(walking(mc1gt_2),232).
orientation(mc1gt_0,147,233). coords(mc1gt_0,163,136,233).  visible(mc1gt_0,233). happensAt(walking(mc1gt_0),233). orientation(mc1gt_2,147,233). coords(mc1gt_2,130,134,233).  visible(mc1gt_2,233). happensAt(walking(mc1gt_2),233).
orientation(mc1gt_0,147,234). coords(mc1gt_0,164,137,234).  visible(mc1gt_0,234). happensAt(walking(mc1gt_0),234). orientation(mc1gt_2,147,234). coords(mc1gt_2,131,134,234).  visible(mc1gt_2,234). happensAt(walking(mc1gt_2),234).
orientation(mc1gt_0,147,235). coords(mc1gt_0,165,138,235).  visible(mc1gt_0,235). happensAt(walking(mc1gt_0),235). orientation(mc1gt_2,147,235). coords(mc1gt_2,131,135,235).  visible(mc1gt_2,235). happensAt(walking(mc1gt_2),235).
orientation(mc1gt_0,147,236). coords(mc1gt_0,165,138,236).  visible(mc1gt_0,236). happensAt(walking(mc1gt_0),236). orientation(mc1gt_2,146,236). coords(mc1gt_2,131,135,236).  visible(mc1gt_2,236). happensAt(walking(mc1gt_2),236).
orientation(mc1gt_0,147,237). coords(mc1gt_0,166,139,237).  visible(mc1gt_0,237). happensAt(walking(mc1gt_0),237). orientation(mc1gt_2,146,237). coords(mc1gt_2,132,135,237).  visible(mc1gt_2,237). happensAt(walking(mc1gt_2),237).
orientation(mc1gt_0,147,238). coords(mc1gt_0,166,139,238).  visible(mc1gt_0,238). happensAt(walking(mc1gt_0),238). orientation(mc1gt_2,146,238). coords(mc1gt_2,132,136,238).  visible(mc1gt_2,238). happensAt(walking(mc1gt_2),238).
orientation(mc1gt_0,147,239). coords(mc1gt_0,166,140,239).  visible(mc1gt_0,239). happensAt(walking(mc1gt_0),239). orientation(mc1gt_2,146,239). coords(mc1gt_2,132,136,239).  visible(mc1gt_2,239). happensAt(walking(mc1gt_2),239).
orientation(mc1gt_0,147,240). coords(mc1gt_0,166,140,240).  visible(mc1gt_0,240). happensAt(walking(mc1gt_0),240). orientation(mc1gt_2,146,240). coords(mc1gt_2,133,137,240).  visible(mc1gt_2,240). happensAt(walking(mc1gt_2),240).
orientation(mc1gt_0,147,241). coords(mc1gt_0,167,143,241).  visible(mc1gt_0,241). happensAt(walking(mc1gt_0),241). orientation(mc1gt_2,146,241). coords(mc1gt_2,133,137,241).  visible(mc1gt_2,241). happensAt(walking(mc1gt_2),241).
orientation(mc1gt_0,147,242). coords(mc1gt_0,167,146,242).  visible(mc1gt_0,242). happensAt(walking(mc1gt_0),242). orientation(mc1gt_2,146,242). coords(mc1gt_2,133,137,242).  visible(mc1gt_2,242). happensAt(walking(mc1gt_2),242).
orientation(mc1gt_0,147,243). coords(mc1gt_0,167,149,243).  visible(mc1gt_0,243). happensAt(walking(mc1gt_0),243). orientation(mc1gt_2,146,243). coords(mc1gt_2,134,137,243).  visible(mc1gt_2,243). happensAt(walking(mc1gt_2),243).
orientation(mc1gt_0,147,244). coords(mc1gt_0,168,151,244).  visible(mc1gt_0,244). happensAt(walking(mc1gt_0),244). orientation(mc1gt_2,146,244). coords(mc1gt_2,134,138,244).  visible(mc1gt_2,244). happensAt(walking(mc1gt_2),244).
orientation(mc1gt_0,143,245). coords(mc1gt_0,169,152,245).  visible(mc1gt_0,245). happensAt(walking(mc1gt_0),245). orientation(mc1gt_2,146,245). coords(mc1gt_2,135,138,245).  visible(mc1gt_2,245). happensAt(walking(mc1gt_2),245).
orientation(mc1gt_0,143,246). coords(mc1gt_0,170,153,246).  visible(mc1gt_0,246). happensAt(walking(mc1gt_0),246). orientation(mc1gt_2,146,246). coords(mc1gt_2,135,139,246).  visible(mc1gt_2,246). happensAt(walking(mc1gt_2),246).
orientation(mc1gt_0,143,247). coords(mc1gt_0,171,155,247).  visible(mc1gt_0,247). happensAt(walking(mc1gt_0),247). orientation(mc1gt_2,146,247). coords(mc1gt_2,136,139,247).  visible(mc1gt_2,247). happensAt(walking(mc1gt_2),247).
orientation(mc1gt_0,143,248). coords(mc1gt_0,171,156,248).  visible(mc1gt_0,248). happensAt(walking(mc1gt_0),248). orientation(mc1gt_2,146,248). coords(mc1gt_2,137,140,248).  visible(mc1gt_2,248). happensAt(walking(mc1gt_2),248).
orientation(mc1gt_0,143,249). coords(mc1gt_0,171,156,249).  visible(mc1gt_0,249). happensAt(walking(mc1gt_0),249). orientation(mc1gt_2,146,249). coords(mc1gt_2,137,140,249).  visible(mc1gt_2,249). happensAt(walking(mc1gt_2),249).
#show holdsAt/2.