
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

	orientation(br1gt_0,175,12). coords(br1gt_0,267,262,12).  visible(br1gt_0,12). happensAt(walking(br1gt_0),12). orientation(br1gt_1,173,12). coords(br1gt_1,187,227,12).  visible(br1gt_1,12). happensAt(walking(br1gt_1),12).
orientation(br1gt_0,175,13). coords(br1gt_0,264,260,13).  visible(br1gt_0,13). happensAt(walking(br1gt_0),13). orientation(br1gt_1,173,13). coords(br1gt_1,186,225,13).  visible(br1gt_1,13). happensAt(walking(br1gt_1),13).
orientation(br1gt_0,6,14). coords(br1gt_0,260,257,14).  visible(br1gt_0,14). happensAt(walking(br1gt_0),14). orientation(br1gt_1,173,14). coords(br1gt_1,185,222,14).  visible(br1gt_1,14). happensAt(walking(br1gt_1),14).
orientation(br1gt_0,173,15). coords(br1gt_0,259,252,15).  visible(br1gt_0,15). happensAt(walking(br1gt_0),15). orientation(br1gt_1,173,15). coords(br1gt_1,185,221,15).  visible(br1gt_1,15). happensAt(walking(br1gt_1),15).
orientation(br1gt_0,169,16). coords(br1gt_0,258,249,16).  visible(br1gt_0,16). happensAt(walking(br1gt_0),16). orientation(br1gt_1,167,16). coords(br1gt_1,184,221,16).  visible(br1gt_1,16). happensAt(walking(br1gt_1),16).
orientation(br1gt_0,165,17). coords(br1gt_0,258,245,17).  visible(br1gt_0,17). happensAt(walking(br1gt_0),17). orientation(br1gt_1,167,17). coords(br1gt_1,182,220,17).  visible(br1gt_1,17). happensAt(walking(br1gt_1),17).
orientation(br1gt_0,164,18). coords(br1gt_0,256,242,18).  visible(br1gt_0,18). happensAt(walking(br1gt_0),18). orientation(br1gt_1,167,18). coords(br1gt_1,180,219,18).  visible(br1gt_1,18). happensAt(walking(br1gt_1),18).
orientation(br1gt_0,162,19). coords(br1gt_0,256,241,19).  visible(br1gt_0,19). happensAt(walking(br1gt_0),19). orientation(br1gt_1,167,19). coords(br1gt_1,178,219,19).  visible(br1gt_1,19). happensAt(walking(br1gt_1),19).
orientation(br1gt_0,159,20). coords(br1gt_0,255,241,20).  visible(br1gt_0,20). happensAt(walking(br1gt_0),20). orientation(br1gt_1,167,20). coords(br1gt_1,174,217,20).  visible(br1gt_1,20). happensAt(walking(br1gt_1),20).
orientation(br1gt_0,159,21). coords(br1gt_0,254,240,21).  visible(br1gt_0,21). happensAt(walking(br1gt_0),21). orientation(br1gt_1,167,21). coords(br1gt_1,172,215,21).  visible(br1gt_1,21). happensAt(walking(br1gt_1),21).
orientation(br1gt_0,164,22). coords(br1gt_0,249,239,22).  visible(br1gt_0,22). happensAt(walking(br1gt_0),22). orientation(br1gt_1,167,22). coords(br1gt_1,168,214,22).  visible(br1gt_1,22). happensAt(walking(br1gt_1),22).
orientation(br1gt_0,169,23). coords(br1gt_0,246,238,23).  visible(br1gt_0,23). happensAt(walking(br1gt_0),23). orientation(br1gt_1,167,23). coords(br1gt_1,166,213,23).  visible(br1gt_1,23). happensAt(walking(br1gt_1),23).
orientation(br1gt_0,174,24). coords(br1gt_0,243,234,24).  visible(br1gt_0,24). happensAt(walking(br1gt_0),24). orientation(br1gt_1,167,24). coords(br1gt_1,165,212,24).  visible(br1gt_1,24). happensAt(walking(br1gt_1),24).
orientation(br1gt_0,172,25). coords(br1gt_0,240,230,25).  visible(br1gt_0,25). happensAt(walking(br1gt_0),25). orientation(br1gt_1,167,25). coords(br1gt_1,165,210,25).  visible(br1gt_1,25). happensAt(walking(br1gt_1),25).
orientation(br1gt_0,172,26). coords(br1gt_0,237,225,26).  visible(br1gt_0,26). happensAt(walking(br1gt_0),26). orientation(br1gt_1,167,26). coords(br1gt_1,163,210,26).  visible(br1gt_1,26). happensAt(walking(br1gt_1),26).
orientation(br1gt_0,172,27). coords(br1gt_0,236,221,27).  visible(br1gt_0,27). happensAt(walking(br1gt_0),27). orientation(br1gt_1,167,27). coords(br1gt_1,163,209,27).  visible(br1gt_1,27). happensAt(walking(br1gt_1),27).
orientation(br1gt_0,172,28). coords(br1gt_0,235,216,28).  visible(br1gt_0,28). happensAt(walking(br1gt_0),28). orientation(br1gt_1,167,28). coords(br1gt_1,162,208,28).  visible(br1gt_1,28). happensAt(walking(br1gt_1),28).
orientation(br1gt_0,172,29). coords(br1gt_0,233,214,29).  visible(br1gt_0,29). happensAt(walking(br1gt_0),29). orientation(br1gt_1,167,29). coords(br1gt_1,161,208,29).  visible(br1gt_1,29). happensAt(walking(br1gt_1),29).
orientation(br1gt_0,172,30). coords(br1gt_0,232,211,30).  visible(br1gt_0,30). happensAt(walking(br1gt_0),30). orientation(br1gt_1,167,30). coords(br1gt_1,160,207,30).  visible(br1gt_1,30). happensAt(walking(br1gt_1),30).
orientation(br1gt_0,170,31). coords(br1gt_0,232,210,31).  visible(br1gt_0,31). happensAt(walking(br1gt_0),31). orientation(br1gt_1,167,31). coords(br1gt_1,158,206,31).  visible(br1gt_1,31). happensAt(walking(br1gt_1),31).
orientation(br1gt_0,170,32). coords(br1gt_0,233,209,32).  visible(br1gt_0,32). happensAt(walking(br1gt_0),32). orientation(br1gt_1,167,32). coords(br1gt_1,156,206,32).  visible(br1gt_1,32). happensAt(walking(br1gt_1),32).
orientation(br1gt_0,170,33). coords(br1gt_0,232,207,33).  visible(br1gt_0,33). happensAt(walking(br1gt_0),33). orientation(br1gt_1,167,33). coords(br1gt_1,153,205,33).  visible(br1gt_1,33). happensAt(walking(br1gt_1),33).
orientation(br1gt_0,166,34). coords(br1gt_0,232,206,34).  visible(br1gt_0,34). happensAt(walking(br1gt_0),34). orientation(br1gt_1,167,34). coords(br1gt_1,150,205,34).  visible(br1gt_1,34). happensAt(walking(br1gt_1),34).
orientation(br1gt_0,164,35). coords(br1gt_0,231,205,35).  visible(br1gt_0,35). happensAt(walking(br1gt_0),35). orientation(br1gt_1,167,35). coords(br1gt_1,147,204,35).  visible(br1gt_1,35). happensAt(walking(br1gt_1),35).
orientation(br1gt_0,162,36). coords(br1gt_0,229,203,36).  visible(br1gt_0,36). happensAt(walking(br1gt_0),36). orientation(br1gt_1,167,36). coords(br1gt_1,144,202,36).  visible(br1gt_1,36). happensAt(walking(br1gt_1),36).
#show holdsAt/2.