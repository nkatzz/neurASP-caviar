
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

	orientation(fra2gt_3,161,214). coords(fra2gt_3,46,168,214).  visible(fra2gt_3,214). happensAt(walking(fra2gt_3),214). orientation(fra2gt_4,7,214). coords(fra2gt_4,212,242,214).  visible(fra2gt_4,214). happensAt(walking(fra2gt_4),214).
orientation(fra2gt_3,161,215). coords(fra2gt_3,46,167,215).  visible(fra2gt_3,215). happensAt(walking(fra2gt_3),215). orientation(fra2gt_4,7,215). coords(fra2gt_4,211,239,215).  visible(fra2gt_4,215). happensAt(walking(fra2gt_4),215).
orientation(fra2gt_3,161,216). coords(fra2gt_3,47,166,216).  visible(fra2gt_3,216). happensAt(walking(fra2gt_3),216). orientation(fra2gt_4,7,216). coords(fra2gt_4,210,235,216).  visible(fra2gt_4,216). happensAt(walking(fra2gt_4),216).
orientation(fra2gt_3,161,217). coords(fra2gt_3,48,166,217).  visible(fra2gt_3,217). happensAt(walking(fra2gt_3),217). orientation(fra2gt_4,7,217). coords(fra2gt_4,209,232,217).  visible(fra2gt_4,217). happensAt(walking(fra2gt_4),217).
orientation(fra2gt_3,161,218). coords(fra2gt_3,49,166,218).  visible(fra2gt_3,218). happensAt(walking(fra2gt_3),218). orientation(fra2gt_4,170,218). coords(fra2gt_4,208,229,218).  visible(fra2gt_4,218). happensAt(walking(fra2gt_4),218).
orientation(fra2gt_3,161,219). coords(fra2gt_3,49,166,219).  visible(fra2gt_3,219). happensAt(walking(fra2gt_3),219). orientation(fra2gt_4,170,219). coords(fra2gt_4,207,228,219).  visible(fra2gt_4,219). happensAt(walking(fra2gt_4),219).
orientation(fra2gt_3,161,220). coords(fra2gt_3,49,165,220).  visible(fra2gt_3,220). happensAt(walking(fra2gt_3),220). orientation(fra2gt_4,170,220). coords(fra2gt_4,207,228,220).  visible(fra2gt_4,220). happensAt(walking(fra2gt_4),220).
orientation(fra2gt_3,161,221). coords(fra2gt_3,50,165,221).  visible(fra2gt_3,221). happensAt(walking(fra2gt_3),221). orientation(fra2gt_4,170,221). coords(fra2gt_4,205,226,221).  visible(fra2gt_4,221). happensAt(walking(fra2gt_4),221).
orientation(fra2gt_3,161,222). coords(fra2gt_3,50,165,222).  visible(fra2gt_3,222). happensAt(walking(fra2gt_3),222). orientation(fra2gt_4,170,222). coords(fra2gt_4,203,225,222).  visible(fra2gt_4,222). happensAt(walking(fra2gt_4),222).
orientation(fra2gt_3,161,223). coords(fra2gt_3,50,164,223).  visible(fra2gt_3,223). happensAt(walking(fra2gt_3),223). orientation(fra2gt_4,170,223). coords(fra2gt_4,201,223,223).  visible(fra2gt_4,223). happensAt(walking(fra2gt_4),223).
orientation(fra2gt_3,161,224). coords(fra2gt_3,50,163,224).  visible(fra2gt_3,224). happensAt(walking(fra2gt_3),224). orientation(fra2gt_4,170,224). coords(fra2gt_4,198,218,224).  visible(fra2gt_4,224). happensAt(walking(fra2gt_4),224).
orientation(fra2gt_3,161,225). coords(fra2gt_3,50,162,225).  visible(fra2gt_3,225). happensAt(walking(fra2gt_3),225). orientation(fra2gt_4,163,225). coords(fra2gt_4,197,214,225).  visible(fra2gt_4,225). happensAt(walking(fra2gt_4),225).
orientation(fra2gt_3,161,226). coords(fra2gt_3,51,161,226).  visible(fra2gt_3,226). happensAt(walking(fra2gt_3),226). orientation(fra2gt_4,163,226). coords(fra2gt_4,197,209,226).  visible(fra2gt_4,226). happensAt(walking(fra2gt_4),226).
orientation(fra2gt_3,161,227). coords(fra2gt_3,51,160,227).  visible(fra2gt_3,227). happensAt(walking(fra2gt_3),227). orientation(fra2gt_4,165,227). coords(fra2gt_4,195,205,227).  visible(fra2gt_4,227). happensAt(walking(fra2gt_4),227).
orientation(fra2gt_3,161,228). coords(fra2gt_3,51,159,228).  visible(fra2gt_3,228). happensAt(walking(fra2gt_3),228). orientation(fra2gt_4,165,228). coords(fra2gt_4,195,203,228).  visible(fra2gt_4,228). happensAt(walking(fra2gt_4),228).
orientation(fra2gt_3,161,229). coords(fra2gt_3,51,159,229).  visible(fra2gt_3,229). happensAt(walking(fra2gt_3),229). orientation(fra2gt_4,159,229). coords(fra2gt_4,194,201,229).  visible(fra2gt_4,229). happensAt(walking(fra2gt_4),229).
orientation(fra2gt_3,161,230). coords(fra2gt_3,53,159,230).  visible(fra2gt_3,230). happensAt(walking(fra2gt_3),230). orientation(fra2gt_4,159,230). coords(fra2gt_4,194,199,230).  visible(fra2gt_4,230). happensAt(walking(fra2gt_4),230).
orientation(fra2gt_3,161,231). coords(fra2gt_3,54,158,231).  visible(fra2gt_3,231). happensAt(walking(fra2gt_3),231). orientation(fra2gt_4,159,231). coords(fra2gt_4,193,197,231).  visible(fra2gt_4,231). happensAt(walking(fra2gt_4),231).
orientation(fra2gt_3,161,232). coords(fra2gt_3,54,158,232).  visible(fra2gt_3,232). happensAt(walking(fra2gt_3),232). orientation(fra2gt_4,159,232). coords(fra2gt_4,192,196,232).  visible(fra2gt_4,232). happensAt(walking(fra2gt_4),232).
orientation(fra2gt_3,161,233). coords(fra2gt_3,55,158,233).  visible(fra2gt_3,233). happensAt(walking(fra2gt_3),233). orientation(fra2gt_4,159,233). coords(fra2gt_4,190,194,233).  visible(fra2gt_4,233). happensAt(walking(fra2gt_4),233).
orientation(fra2gt_3,161,234). coords(fra2gt_3,55,157,234).  visible(fra2gt_3,234). happensAt(walking(fra2gt_3),234). orientation(fra2gt_4,159,234). coords(fra2gt_4,189,191,234).  visible(fra2gt_4,234). happensAt(walking(fra2gt_4),234).
orientation(fra2gt_3,161,235). coords(fra2gt_3,55,156,235).  visible(fra2gt_3,235). happensAt(walking(fra2gt_3),235). orientation(fra2gt_4,159,235). coords(fra2gt_4,188,189,235).  visible(fra2gt_4,235). happensAt(walking(fra2gt_4),235).
orientation(fra2gt_3,161,236). coords(fra2gt_3,55,156,236).  visible(fra2gt_3,236). happensAt(walking(fra2gt_3),236). orientation(fra2gt_4,159,236). coords(fra2gt_4,184,187,236).  visible(fra2gt_4,236). happensAt(walking(fra2gt_4),236).
orientation(fra2gt_3,156,237). coords(fra2gt_3,55,155,237).  visible(fra2gt_3,237). happensAt(walking(fra2gt_3),237). orientation(fra2gt_4,159,237). coords(fra2gt_4,182,185,237).  visible(fra2gt_4,237). happensAt(walking(fra2gt_4),237).
orientation(fra2gt_3,151,238). coords(fra2gt_3,55,153,238).  visible(fra2gt_3,238). happensAt(walking(fra2gt_3),238). orientation(fra2gt_4,159,238). coords(fra2gt_4,179,183,238).  visible(fra2gt_4,238). happensAt(walking(fra2gt_4),238).
#show holdsAt/2.