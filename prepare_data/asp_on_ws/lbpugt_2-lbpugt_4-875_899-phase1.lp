
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
orientation(lbpugt_2,141,888). coords(lbpugt_2,267,120,888).  visible(lbpugt_2,888). happensAt(walking(lbpugt_2),888). orientation(lbpugt_4,135,888). coords(lbpugt_4,219,34,888).  visible(lbpugt_4,888). happensAt(inactive(lbpugt_4),888).
orientation(lbpugt_2,141,889). coords(lbpugt_2,265,121,889).  visible(lbpugt_2,889). happensAt(walking(lbpugt_2),889). orientation(lbpugt_4,135,889). coords(lbpugt_4,219,34,889).  visible(lbpugt_4,889). happensAt(inactive(lbpugt_4),889).
orientation(lbpugt_2,146,890). coords(lbpugt_2,263,121,890).  visible(lbpugt_2,890). happensAt(walking(lbpugt_2),890). orientation(lbpugt_4,135,890). coords(lbpugt_4,219,34,890).  visible(lbpugt_4,890). happensAt(inactive(lbpugt_4),890).
orientation(lbpugt_2,146,891). coords(lbpugt_2,262,121,891).  visible(lbpugt_2,891). happensAt(walking(lbpugt_2),891). orientation(lbpugt_4,135,891). coords(lbpugt_4,219,34,891).  visible(lbpugt_4,891). happensAt(inactive(lbpugt_4),891).
orientation(lbpugt_2,146,892). coords(lbpugt_2,259,122,892).  visible(lbpugt_2,892). happensAt(walking(lbpugt_2),892). orientation(lbpugt_4,135,892). coords(lbpugt_4,219,34,892).  visible(lbpugt_4,892). happensAt(inactive(lbpugt_4),892).
orientation(lbpugt_2,146,893). coords(lbpugt_2,257,122,893).  visible(lbpugt_2,893). happensAt(walking(lbpugt_2),893). orientation(lbpugt_4,135,893). coords(lbpugt_4,219,34,893).  visible(lbpugt_4,893). happensAt(inactive(lbpugt_4),893).
orientation(lbpugt_2,142,894). coords(lbpugt_2,254,122,894).  visible(lbpugt_2,894). happensAt(walking(lbpugt_2),894). orientation(lbpugt_4,135,894). coords(lbpugt_4,219,34,894).  visible(lbpugt_4,894). happensAt(inactive(lbpugt_4),894).
orientation(lbpugt_2,136,895). coords(lbpugt_2,252,123,895).  visible(lbpugt_2,895). happensAt(walking(lbpugt_2),895). orientation(lbpugt_4,135,895). coords(lbpugt_4,219,34,895).  visible(lbpugt_4,895). happensAt(inactive(lbpugt_4),895).
orientation(lbpugt_2,136,896). coords(lbpugt_2,251,123,896).  visible(lbpugt_2,896). happensAt(walking(lbpugt_2),896). orientation(lbpugt_4,135,896). coords(lbpugt_4,219,34,896).  visible(lbpugt_4,896). happensAt(inactive(lbpugt_4),896).
orientation(lbpugt_2,136,897). coords(lbpugt_2,250,124,897).  visible(lbpugt_2,897). happensAt(walking(lbpugt_2),897). orientation(lbpugt_4,135,897). coords(lbpugt_4,219,34,897).  visible(lbpugt_4,897). happensAt(inactive(lbpugt_4),897).
orientation(lbpugt_2,136,898). coords(lbpugt_2,250,124,898).  visible(lbpugt_2,898). happensAt(walking(lbpugt_2),898). orientation(lbpugt_4,135,898). coords(lbpugt_4,219,34,898).  visible(lbpugt_4,898). happensAt(inactive(lbpugt_4),898).
orientation(lbpugt_2,143,899). coords(lbpugt_2,249,124,899).  visible(lbpugt_2,899). happensAt(walking(lbpugt_2),899). orientation(lbpugt_4,135,899). coords(lbpugt_4,219,34,899).  visible(lbpugt_4,899). happensAt(inactive(lbpugt_4),899).
#show holdsAt/2.