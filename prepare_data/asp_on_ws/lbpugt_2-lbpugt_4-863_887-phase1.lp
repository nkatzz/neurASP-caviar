
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

	orientation(lbpugt_2,87,863). coords(lbpugt_2,290,101,863).  visible(lbpugt_2,863). happensAt(walking(lbpugt_2),863). orientation(lbpugt_4,135,863). coords(lbpugt_4,219,34,863).  visible(lbpugt_4,863). happensAt(inactive(lbpugt_4),863).
orientation(lbpugt_2,89,864). coords(lbpugt_2,290,101,864).  visible(lbpugt_2,864). happensAt(walking(lbpugt_2),864). orientation(lbpugt_4,135,864). coords(lbpugt_4,219,34,864).  visible(lbpugt_4,864). happensAt(inactive(lbpugt_4),864).
orientation(lbpugt_2,92,865). coords(lbpugt_2,289,102,865).  visible(lbpugt_2,865). happensAt(walking(lbpugt_2),865). orientation(lbpugt_4,135,865). coords(lbpugt_4,219,34,865).  visible(lbpugt_4,865). happensAt(inactive(lbpugt_4),865).
orientation(lbpugt_2,93,866). coords(lbpugt_2,289,103,866).  visible(lbpugt_2,866). happensAt(walking(lbpugt_2),866). orientation(lbpugt_4,135,866). coords(lbpugt_4,219,34,866).  visible(lbpugt_4,866). happensAt(inactive(lbpugt_4),866).
orientation(lbpugt_2,100,867). coords(lbpugt_2,289,103,867).  visible(lbpugt_2,867). happensAt(walking(lbpugt_2),867). orientation(lbpugt_4,135,867). coords(lbpugt_4,219,34,867).  visible(lbpugt_4,867). happensAt(inactive(lbpugt_4),867).
orientation(lbpugt_2,100,868). coords(lbpugt_2,288,104,868).  visible(lbpugt_2,868). happensAt(walking(lbpugt_2),868). orientation(lbpugt_4,135,868). coords(lbpugt_4,219,34,868).  visible(lbpugt_4,868). happensAt(inactive(lbpugt_4),868).
orientation(lbpugt_2,109,869). coords(lbpugt_2,288,105,869).  visible(lbpugt_2,869). happensAt(walking(lbpugt_2),869). orientation(lbpugt_4,135,869). coords(lbpugt_4,219,34,869).  visible(lbpugt_4,869). happensAt(inactive(lbpugt_4),869).
orientation(lbpugt_2,109,870). coords(lbpugt_2,287,105,870).  visible(lbpugt_2,870). happensAt(walking(lbpugt_2),870). orientation(lbpugt_4,135,870). coords(lbpugt_4,219,34,870).  visible(lbpugt_4,870). happensAt(inactive(lbpugt_4),870).
orientation(lbpugt_2,116,871). coords(lbpugt_2,286,107,871).  visible(lbpugt_2,871). happensAt(walking(lbpugt_2),871). orientation(lbpugt_4,135,871). coords(lbpugt_4,219,34,871).  visible(lbpugt_4,871). happensAt(inactive(lbpugt_4),871).
orientation(lbpugt_2,116,872). coords(lbpugt_2,286,107,872).  visible(lbpugt_2,872). happensAt(walking(lbpugt_2),872). orientation(lbpugt_4,135,872). coords(lbpugt_4,219,34,872).  visible(lbpugt_4,872). happensAt(inactive(lbpugt_4),872).
orientation(lbpugt_2,123,873). coords(lbpugt_2,285,108,873).  visible(lbpugt_2,873). happensAt(walking(lbpugt_2),873). orientation(lbpugt_4,135,873). coords(lbpugt_4,219,34,873).  visible(lbpugt_4,873). happensAt(inactive(lbpugt_4),873).
orientation(lbpugt_2,123,874). coords(lbpugt_2,285,108,874).  visible(lbpugt_2,874). happensAt(walking(lbpugt_2),874). orientation(lbpugt_4,135,874). coords(lbpugt_4,219,34,874).  visible(lbpugt_4,874). happensAt(inactive(lbpugt_4),874).
orientation(lbpugt_2,128,875). coords(lbpugt_2,284,109,875).  visible(lbpugt_2,875). happensAt(walking(lbpugt_2),875). orientation(lbpugt_4,135,875). coords(lbpugt_4,219,34,875).  visible(lbpugt_4,875). happensAt(inactive(lbpugt_4),875).
orientation(lbpugt_2,128,876). coords(lbpugt_2,283,110,876).  visible(lbpugt_2,876). happensAt(walking(lbpugt_2),876). orientation(lbpugt_4,135,876). coords(lbpugt_4,219,34,876).  visible(lbpugt_4,876). happensAt(inactive(lbpugt_4),876).
orientation(lbpugt_2,128,877). coords(lbpugt_2,281,110,877).  visible(lbpugt_2,877). happensAt(walking(lbpugt_2),877). orientation(lbpugt_4,135,877). coords(lbpugt_4,219,34,877).  visible(lbpugt_4,877). happensAt(inactive(lbpugt_4),877).
orientation(lbpugt_2,128,878). coords(lbpugt_2,280,110,878).  visible(lbpugt_2,878). happensAt(walking(lbpugt_2),878). orientation(lbpugt_4,135,878). coords(lbpugt_4,219,34,878).  visible(lbpugt_4,878). happensAt(inactive(lbpugt_4),878).
orientation(lbpugt_2,135,879). coords(lbpugt_2,277,111,879).  visible(lbpugt_2,879). happensAt(walking(lbpugt_2),879). orientation(lbpugt_4,135,879). coords(lbpugt_4,219,34,879).  visible(lbpugt_4,879). happensAt(inactive(lbpugt_4),879).
orientation(lbpugt_2,124,880). coords(lbpugt_2,275,114,880).  visible(lbpugt_2,880). happensAt(walking(lbpugt_2),880). orientation(lbpugt_4,135,880). coords(lbpugt_4,219,34,880).  visible(lbpugt_4,880). happensAt(inactive(lbpugt_4),880).
orientation(lbpugt_2,124,881). coords(lbpugt_2,274,115,881).  visible(lbpugt_2,881). happensAt(walking(lbpugt_2),881). orientation(lbpugt_4,135,881). coords(lbpugt_4,219,34,881).  visible(lbpugt_4,881). happensAt(inactive(lbpugt_4),881).
orientation(lbpugt_2,124,882). coords(lbpugt_2,273,117,882).  visible(lbpugt_2,882). happensAt(walking(lbpugt_2),882). orientation(lbpugt_4,135,882). coords(lbpugt_4,219,34,882).  visible(lbpugt_4,882). happensAt(inactive(lbpugt_4),882).
orientation(lbpugt_2,118,883). coords(lbpugt_2,272,119,883).  visible(lbpugt_2,883). happensAt(walking(lbpugt_2),883). orientation(lbpugt_4,135,883). coords(lbpugt_4,219,34,883).  visible(lbpugt_4,883). happensAt(inactive(lbpugt_4),883).
orientation(lbpugt_2,127,884). coords(lbpugt_2,271,119,884).  visible(lbpugt_2,884). happensAt(walking(lbpugt_2),884). orientation(lbpugt_4,135,884). coords(lbpugt_4,219,34,884).  visible(lbpugt_4,884). happensAt(inactive(lbpugt_4),884).
orientation(lbpugt_2,127,885). coords(lbpugt_2,270,119,885).  visible(lbpugt_2,885). happensAt(walking(lbpugt_2),885). orientation(lbpugt_4,135,885). coords(lbpugt_4,219,34,885).  visible(lbpugt_4,885). happensAt(inactive(lbpugt_4),885).
orientation(lbpugt_2,134,886). coords(lbpugt_2,269,119,886).  visible(lbpugt_2,886). happensAt(walking(lbpugt_2),886). orientation(lbpugt_4,135,886). coords(lbpugt_4,219,34,886).  visible(lbpugt_4,886). happensAt(inactive(lbpugt_4),886).
orientation(lbpugt_2,134,887). coords(lbpugt_2,268,119,887).  visible(lbpugt_2,887). happensAt(walking(lbpugt_2),887). orientation(lbpugt_4,135,887). coords(lbpugt_4,219,34,887).  visible(lbpugt_4,887). happensAt(inactive(lbpugt_4),887).
#show holdsAt/2.