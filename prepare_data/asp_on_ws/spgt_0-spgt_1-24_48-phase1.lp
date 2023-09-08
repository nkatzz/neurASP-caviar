
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

	orientation(spgt_0,156,24). coords(spgt_0,41,167,24).  visible(spgt_0,24). happensAt(walking(spgt_0),24). orientation(spgt_1,160,24). coords(spgt_1,41,158,24).  visible(spgt_1,24). happensAt(walking(spgt_1),24).
orientation(spgt_0,156,25). coords(spgt_0,42,167,25).  visible(spgt_0,25). happensAt(walking(spgt_0),25). orientation(spgt_1,160,25). coords(spgt_1,41,158,25).  visible(spgt_1,25). happensAt(walking(spgt_1),25).
orientation(spgt_0,156,26). coords(spgt_0,42,167,26).  visible(spgt_0,26). happensAt(walking(spgt_0),26). orientation(spgt_1,160,26). coords(spgt_1,41,158,26).  visible(spgt_1,26). happensAt(walking(spgt_1),26).
orientation(spgt_0,156,27). coords(spgt_0,42,166,27).  visible(spgt_0,27). happensAt(walking(spgt_0),27). orientation(spgt_1,160,27). coords(spgt_1,42,158,27).  visible(spgt_1,27). happensAt(walking(spgt_1),27).
orientation(spgt_0,156,28). coords(spgt_0,42,166,28).  visible(spgt_0,28). happensAt(walking(spgt_0),28). orientation(spgt_1,160,28). coords(spgt_1,42,158,28).  visible(spgt_1,28). happensAt(walking(spgt_1),28).
orientation(spgt_0,156,29). coords(spgt_0,42,166,29).  visible(spgt_0,29). happensAt(walking(spgt_0),29). orientation(spgt_1,160,29). coords(spgt_1,43,158,29).  visible(spgt_1,29). happensAt(walking(spgt_1),29).
orientation(spgt_0,156,30). coords(spgt_0,42,166,30).  visible(spgt_0,30). happensAt(walking(spgt_0),30). orientation(spgt_1,160,30). coords(spgt_1,43,158,30).  visible(spgt_1,30). happensAt(walking(spgt_1),30).
orientation(spgt_0,156,31). coords(spgt_0,42,166,31).  visible(spgt_0,31). happensAt(walking(spgt_0),31). orientation(spgt_1,160,31). coords(spgt_1,43,158,31).  visible(spgt_1,31). happensAt(walking(spgt_1),31).
orientation(spgt_0,156,32). coords(spgt_0,42,166,32).  visible(spgt_0,32). happensAt(walking(spgt_0),32). orientation(spgt_1,160,32). coords(spgt_1,44,158,32).  visible(spgt_1,32). happensAt(walking(spgt_1),32).
orientation(spgt_0,156,33). coords(spgt_0,42,166,33).  visible(spgt_0,33). happensAt(walking(spgt_0),33). orientation(spgt_1,160,33). coords(spgt_1,44,158,33).  visible(spgt_1,33). happensAt(walking(spgt_1),33).
orientation(spgt_0,156,34). coords(spgt_0,42,166,34).  visible(spgt_0,34). happensAt(walking(spgt_0),34). orientation(spgt_1,160,34). coords(spgt_1,45,157,34).  visible(spgt_1,34). happensAt(walking(spgt_1),34).
orientation(spgt_0,156,35). coords(spgt_0,44,166,35).  visible(spgt_0,35). happensAt(walking(spgt_0),35). orientation(spgt_1,160,35). coords(spgt_1,45,157,35).  visible(spgt_1,35). happensAt(walking(spgt_1),35).
orientation(spgt_0,156,36). coords(spgt_0,45,166,36).  visible(spgt_0,36). happensAt(walking(spgt_0),36). orientation(spgt_1,160,36). coords(spgt_1,45,157,36).  visible(spgt_1,36). happensAt(walking(spgt_1),36).
orientation(spgt_0,156,37). coords(spgt_0,45,166,37).  visible(spgt_0,37). happensAt(walking(spgt_0),37). orientation(spgt_1,160,37). coords(spgt_1,45,157,37).  visible(spgt_1,37). happensAt(walking(spgt_1),37).
orientation(spgt_0,156,38). coords(spgt_0,45,166,38).  visible(spgt_0,38). happensAt(walking(spgt_0),38). orientation(spgt_1,160,38). coords(spgt_1,45,157,38).  visible(spgt_1,38). happensAt(walking(spgt_1),38).
orientation(spgt_0,156,39). coords(spgt_0,45,166,39).  visible(spgt_0,39). happensAt(walking(spgt_0),39). orientation(spgt_1,160,39). coords(spgt_1,46,157,39).  visible(spgt_1,39). happensAt(walking(spgt_1),39).
orientation(spgt_0,156,40). coords(spgt_0,45,166,40).  visible(spgt_0,40). happensAt(walking(spgt_0),40). orientation(spgt_1,160,40). coords(spgt_1,46,157,40).  visible(spgt_1,40). happensAt(walking(spgt_1),40).
orientation(spgt_0,156,41). coords(spgt_0,46,166,41).  visible(spgt_0,41). happensAt(walking(spgt_0),41). orientation(spgt_1,160,41). coords(spgt_1,47,157,41).  visible(spgt_1,41). happensAt(walking(spgt_1),41).
orientation(spgt_0,156,42). coords(spgt_0,46,166,42).  visible(spgt_0,42). happensAt(walking(spgt_0),42). orientation(spgt_1,160,42). coords(spgt_1,47,157,42).  visible(spgt_1,42). happensAt(walking(spgt_1),42).
orientation(spgt_0,156,43). coords(spgt_0,46,165,43).  visible(spgt_0,43). happensAt(walking(spgt_0),43). orientation(spgt_1,160,43). coords(spgt_1,47,157,43).  visible(spgt_1,43). happensAt(walking(spgt_1),43).
orientation(spgt_0,156,44). coords(spgt_0,46,165,44).  visible(spgt_0,44). happensAt(walking(spgt_0),44). orientation(spgt_1,160,44). coords(spgt_1,47,156,44).  visible(spgt_1,44). happensAt(walking(spgt_1),44).
orientation(spgt_0,156,45). coords(spgt_0,46,165,45).  visible(spgt_0,45). happensAt(walking(spgt_0),45). orientation(spgt_1,160,45). coords(spgt_1,47,156,45).  visible(spgt_1,45). happensAt(walking(spgt_1),45).
orientation(spgt_0,156,46). coords(spgt_0,46,165,46).  visible(spgt_0,46). happensAt(walking(spgt_0),46). orientation(spgt_1,160,46). coords(spgt_1,48,156,46).  visible(spgt_1,46). happensAt(walking(spgt_1),46).
orientation(spgt_0,156,47). coords(spgt_0,47,164,47).  visible(spgt_0,47). happensAt(walking(spgt_0),47). orientation(spgt_1,160,47). coords(spgt_1,48,156,47).  visible(spgt_1,47). happensAt(walking(spgt_1),47).
orientation(spgt_0,156,48). coords(spgt_0,49,164,48).  visible(spgt_0,48). happensAt(walking(spgt_0),48). orientation(spgt_1,160,48). coords(spgt_1,48,156,48).  visible(spgt_1,48). happensAt(walking(spgt_1),48).
#show holdsAt/2.